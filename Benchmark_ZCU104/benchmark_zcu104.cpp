// ============================================================================
// Methane segmentation - generic ZCU104 optimized benchmark
// Vitis AI 3.5 / VART / DPUCZDX8G
//
// BATCH IS ALWAYS 1.
//
// Profiles:
//   baseline:
//       - sequential batch=1
//       - 1 runner
//       - 1 inference in flight
//       - reports baseline_model_only + baseline_end_to_end
//
//   max-model-only:
//       - inputs already preprocessed/quantized before timing
//       - N independent VART runners, one host thread per runner
//       - batch=1 for every inference
//       - measures sustained throughput = completed / wall time
//
//   max-e2e:
//       - threaded pipeline:
//             PRE workers -> per-runner DPU queues -> POST workers
//       - multiple independent aligned slots/buffers
//       - disk TIFF I/O INCLUDED
//       - normalization + INT8 quantization INCLUDED
//       - VART sync + DPU INCLUDED
//       - threshold + cross morphological opening INCLUDED
//       - labels/metrics/CSV NOT INCLUDED in timed pipeline
//
// Accuracy validation:
//   TP, FP, FN, TN, Precision, Recall, F1 global/strong/weak, IoU,
//   AUPRC, no-plume FPR and Accuracy
//
// Statistical output:
//   throughput FPS (wall)
//   latency mean / median / min / max / p90 / p95 / p99
//   stddev / CV / p99-p50 jitter
//   inter-completion interval mean/p95/p99
//
// Build command is provided in build_zcu104.sh.
//
// ============================================================================

#include <vart/runner.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/tensor/tensor.hpp>
#include <vitis/ai/graph_runner.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static constexpr int kBatch = 1;
static constexpr int kH = 512;
static constexpr int kW = 512;
static constexpr int kC = 4;

static constexpr int kFallbackInputFix = 5;
static constexpr int kFallbackOutputFix = 2;

// ============================================================================
// OPTIONS
// ============================================================================

struct Options {
  std::string profile = "all";  // all|baseline|max-model-only|max-e2e

  std::string model;
  std::string dataset;
  std::string csv;
  std::string out = "benchmark_results";

  int runners = 2;
  int pre_workers = 2;
  int post_workers = 1;
  int slots_per_runner = 3;

  int iterations = 500;
  int warmup = 20;
  int baseline_repeats = 100;
  int baseline_e2e_passes = 5;
  int samples = 0;
  int cpu_cores = 0;

  bool pin = false;
  bool validate = true;
  bool power = true;
  int power_sample_ms = 200;
  bool self_test = false;
};

static void usage(const char* argv0) {
  std::cout
      << "Methane segmentation ZCU104 benchmark - batch=1 always\n\n"
      << "Usage:\n"
      << "  " << argv0 << " --profile all|baseline|max-model-only|max-e2e [options]\n\n"
      << "Profile aliases:\n"
      << "  --baseline\n"
      << "  --maxthroughputmodelonly\n"
      << "  --maxendend\n\n"
      << "Core:\n"
      << "  --model PATH\n"
      << "  --dataset DIR\n"
      << "  --csv PATH\n"
      << "  --out DIR\n\n"
      << "  --samples N             0=all CSV samples\n"
      << "  --cpu-cores N|--threads N  0=all available, otherwise 1..4\n\n"
      << "Concurrency:\n"
      << "  --runners N\n"
      << "  --pre-workers N\n"
      << "  --post-workers N\n"
      << "  --slots-per-runner N\n"
      << "  --pin | --no-pin\n\n"
      << "Benchmark:\n"
      << "  --iterations N          0=one inference per selected image\n"
      << "  --warmup N\n"
      << "  --baseline-repeats N\n"
      << "  --baseline-e2e-passes N\n"
      << "  --validate | --no-validate\n"
      << "  --power | --no-power   sample ZCU104 hwmon power rails\n"
      << "  --power-sample-ms N    sensor interval (default 200 ms)\n"
      << "  --self-test\n";
}

static Options parse_options(int argc, char** argv) {
  Options o;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];

    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + a);
      }
      return argv[++i];
    };

    if (a == "--profile") o.profile = value();
    else if (a == "--baseline") o.profile = "baseline";
    else if (a == "--maxthroughputmodelonly") o.profile = "max-model-only";
    else if (a == "--maxendend") o.profile = "max-e2e";

    else if (a == "--model") o.model = value();
    else if (a == "--dataset") o.dataset = value();
    else if (a == "--csv") o.csv = value();
    else if (a == "--out") o.out = value();
    else if (a == "--samples") o.samples = std::stoi(value());
    else if (a == "--cpu-cores") o.cpu_cores = std::stoi(value());
    else if (a == "--threads") o.cpu_cores = std::stoi(value());

    else if (a == "--runners") o.runners = std::stoi(value());
    else if (a == "--pre-workers") o.pre_workers = std::stoi(value());
    else if (a == "--post-workers") o.post_workers = std::stoi(value());
    else if (a == "--slots-per-runner") o.slots_per_runner = std::stoi(value());

    else if (a == "--iterations") o.iterations = std::stoi(value());
    else if (a == "--warmup") o.warmup = std::stoi(value());
    else if (a == "--baseline-repeats") o.baseline_repeats = std::stoi(value());
    else if (a == "--baseline-e2e-passes") o.baseline_e2e_passes = std::stoi(value());

    else if (a == "--pin") o.pin = true;
    else if (a == "--no-pin") o.pin = false;

    else if (a == "--validate") o.validate = true;
    else if (a == "--no-validate") o.validate = false;
    else if (a == "--power") o.power = true;
    else if (a == "--no-power") o.power = false;
    else if (a == "--power-sample-ms") o.power_sample_ms = std::stoi(value());
    else if (a == "--self-test") o.self_test = true;

    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (o.profile != "all" &&
      o.profile != "baseline" &&
      o.profile != "max-model-only" &&
      o.profile != "max-e2e") {
    throw std::runtime_error(
        "--profile must be all, baseline, max-model-only or max-e2e");
  }

  if (o.runners <= 0 ||
      o.pre_workers <= 0 ||
      o.post_workers <= 0 ||
      o.slots_per_runner <= 0 ||
      o.iterations < 0 ||
      o.warmup < 0 ||
      o.baseline_repeats <= 0 ||
      o.baseline_e2e_passes <= 0 ||
      o.samples < 0 ||
      o.power_sample_ms <= 0 ||
      o.cpu_cores < 0 || o.cpu_cores > 4) {
    throw std::runtime_error("Invalid numeric option");
  }

  if (!o.self_test && (o.model.empty() || o.dataset.empty())) {
    throw std::runtime_error("--model and --dataset are required");
  }
  if (!o.self_test && o.csv.empty()) o.csv = (fs::path(o.dataset) / "test.csv").string();

  return o;
}

// ============================================================================
// HELPERS
// ============================================================================

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static void ensure_dir(const fs::path& p) {
  fs::create_directories(p);
}

struct PowerRail {
  std::string name;
  fs::path input;
  std::vector<double> watts;
};

class PowerMonitor {
 public:
  explicit PowerMonitor(const Options& o) : interval_ms_(o.power_sample_ms) {
    if (!o.power) return;
    const fs::path root("/sys/class/hwmon");
    if (!fs::exists(root)) return;
    for (const auto& hwmon : fs::directory_iterator(root)) {
      const std::string chip = read_text(hwmon.path() / "name");
      for (const auto& entry : fs::directory_iterator(hwmon.path())) {
        const std::string file = entry.path().filename().string();
        if (file.rfind("power", 0) != 0 || file.size() < 7 ||
            file.substr(file.size() - 6) != "_input") continue;
        const std::string stem = file.substr(0, file.size() - 6);
        const std::string label = read_text(hwmon.path() / (stem + "_label"));
        rails_.push_back({label.empty() ? chip + ":" + stem : label, entry.path(), {}});
      }
    }
  }

  ~PowerMonitor() { stop(); }

  void start() {
    if (rails_.empty()) return;
    stop_ = false;
    sample();
    thread_ = std::thread([this] {
      while (!stop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        if (!stop_.load()) sample();
      }
    });
  }

  void stop() {
    if (!thread_.joinable()) return;
    stop_ = true;
    thread_.join();
    sample();
  }

  std::vector<PowerRail> take() { return std::move(rails_); }

 private:
  static std::string read_text(const fs::path& path) {
    std::ifstream f(path);
    std::string value;
    if (f) std::getline(f, value);
    return value;
  }

  void sample() {
    for (auto& rail : rails_) {
      std::ifstream f(rail.input);
      double microwatts = 0.0;
      if (f >> microwatts) rail.watts.push_back(microwatts / 1000000.0);
    }
  }

  int interval_ms_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::vector<PowerRail> rails_;
};

static std::vector<int> available_cpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
  }
  return cpus;
}

static void limit_process_to_cpus(int count) {
  if (count == 0) return;
  const auto cpus = available_cpus();
  if (count > static_cast<int>(cpus.size())) {
    throw std::runtime_error("--cpu-cores exceeds available CPUs");
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int i = 0; i < count; ++i) CPU_SET(cpus[static_cast<size_t>(i)], &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_setaffinity failed");
  }
}

static void pin_current_thread(int logical_slot) {
  const auto cpus = available_cpus();
  const int core = cpus[static_cast<size_t>(logical_slot) % cpus.size()];

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);

  int rc = pthread_setaffinity_np(
      pthread_self(),
      sizeof(set),
      &set);

  if (rc != 0) {
    std::cerr
        << "WARN: pthread_setaffinity_np(core="
        << core
        << ") failed rc="
        << rc
        << "\n";
  }
}

// ============================================================================
// CSV DATASET
// ============================================================================

struct DatasetItem {
  std::string id;
  fs::path folder;
  std::string has_plume;
  double qplume = 0.0;
};

static std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string cur;
  bool quoted = false;

  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];

    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        cur += '"';
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(cur);
      cur.clear();
    } else {
      cur += ch;
    }
  }

  fields.push_back(cur);
  return fields;
}

static std::vector<DatasetItem> load_dataset_csv(
    const std::string& path,
    const std::string& dataset_root,
    int sample_limit) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open CSV: " + path);

  std::string line;
  if (!std::getline(f, line)) {
    throw std::runtime_error("Empty CSV: " + path);
  }

  auto header = parse_csv_line(line);

  int id_col = -1;
  int folder_col = -1;
  int plume_col = -1;
  int qplume_col = -1;

  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == "id") id_col = static_cast<int>(i);
    if (header[i] == "folder") folder_col = static_cast<int>(i);
    if (header[i] == "has_plume") plume_col = static_cast<int>(i);
    if (header[i] == "qplume") qplume_col = static_cast<int>(i);
  }

  if (id_col < 0 && folder_col < 0) {
    throw std::runtime_error("CSV has neither id nor folder column");
  }

  std::vector<DatasetItem> out;

  while (std::getline(f, line)) {
    if (line.empty()) continue;

    auto fields = parse_csv_line(line);

    DatasetItem item;
    std::string folder_value;
    if (folder_col >= 0 && static_cast<size_t>(folder_col) < fields.size()) {
      folder_value = fields[static_cast<size_t>(folder_col)];
    }
    if (id_col >= 0 && static_cast<size_t>(id_col) < fields.size()) {
      item.id = fields[static_cast<size_t>(id_col)];
    } else {
      item.id = fs::path(folder_value).filename().string();
    }

    const fs::path root(dataset_root);
    const fs::path by_id = root / item.id;
    const fs::path raw(folder_value);
    const fs::path relative = raw.is_absolute() ? root / raw.relative_path() : root / raw;
    const fs::path by_basename = root / raw.filename();
    if (fs::is_directory(by_id)) item.folder = by_id;
    else if (!folder_value.empty() && fs::is_directory(relative)) item.folder = relative;
    else if (!folder_value.empty() && fs::is_directory(by_basename)) item.folder = by_basename;
    else continue;

    if (plume_col >= 0 &&
        static_cast<size_t>(plume_col) < fields.size()) {
      item.has_plume = fields[plume_col];
    }
    if (qplume_col >= 0 && static_cast<size_t>(qplume_col) < fields.size() &&
        !fields[static_cast<size_t>(qplume_col)].empty()) {
      item.qplume = std::stod(fields[static_cast<size_t>(qplume_col)]);
    }

    out.push_back(std::move(item));
    if (sample_limit > 0 && static_cast<int>(out.size()) >= sample_limit) break;
  }

  if (out.empty()) {
    throw std::runtime_error("No samples in CSV");
  }

  return out;
}

// ============================================================================
// ALIGNED STORAGE / TENSOR BUFFER
// ============================================================================

class AlignedStorage {
 public:
  AlignedStorage() = default;

  explicit AlignedStorage(size_t bytes) {
    allocate(bytes);
  }

  AlignedStorage(const AlignedStorage&) = delete;
  AlignedStorage& operator=(const AlignedStorage&) = delete;

  AlignedStorage(AlignedStorage&&) noexcept = default;
  AlignedStorage& operator=(AlignedStorage&&) noexcept = default;

  void allocate(size_t bytes) {
    bytes_ = bytes;

    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes_) != 0 || p == nullptr) {
      throw std::bad_alloc();
    }

    ptr_.reset(static_cast<uint8_t*>(p));
    std::memset(ptr_.get(), 0, bytes_);
  }

  uint8_t* data() { return ptr_.get(); }
  const uint8_t* data() const { return ptr_.get(); }
  size_t size() const { return bytes_; }

 private:
  struct Free {
    void operator()(uint8_t* p) const {
      std::free(p);
    }
  };

  std::unique_ptr<uint8_t, Free> ptr_;
  size_t bytes_ = 0;
};

class CpuFlatTensorBuffer final : public vart::TensorBuffer {
 public:
  CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : vart::TensorBuffer(tensor),
        data_(static_cast<uint8_t*>(data)) {}

  std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
    const auto shape = tensor_->get_shape();

    size_t offset = 0;

    if (!idx.empty()) {
      if (idx.size() != shape.size()) {
        throw std::runtime_error("TensorBuffer rank mismatch");
      }

      for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= shape[i]) {
          throw std::runtime_error("TensorBuffer index out of range");
        }

        offset =
            offset * static_cast<size_t>(shape[i])
            + static_cast<size_t>(idx[i]);
      }
    }

    const size_t total_bytes = tensor_->get_data_size();
    const size_t elements =
        static_cast<size_t>(tensor_->get_element_num());

    const size_t element_bytes =
        total_bytes / elements;

    const size_t byte_offset =
        offset * element_bytes;

    return {
        reinterpret_cast<uint64_t>(data_ + byte_offset),
        total_bytes - byte_offset
    };
  }

 private:
  uint8_t* data_;
};

struct OwnedTensorBuffer {
  explicit OwnedTensorBuffer(const xir::Tensor* tensor)
      : storage(tensor->get_data_size()),
        buffer(storage.data(), tensor) {}

  AlignedStorage storage;
  CpuFlatTensorBuffer buffer;
};

// ============================================================================
// MODEL / RUNNER
// ============================================================================

static void count_device_subgraphs(
    const xir::Subgraph* subgraph,
    int& dpu_count,
    int& cpu_count,
    const xir::Subgraph*& first_dpu) {
  const auto children = subgraph->children_topological_sort();
  if (subgraph->has_attr("device")) {
    const auto device = subgraph->get_attr<std::string>("device");
    if (device == "DPU") {
      ++dpu_count;
      if (first_dpu == nullptr) first_dpu = subgraph;
    }
    else if (device == "CPU") ++cpu_count;
  }
  for (const auto* child : children) {
    count_device_subgraphs(child, dpu_count, cpu_count, first_dpu);
  }
}

struct ModelContext {
  explicit ModelContext(const std::string& path) {
    graph = xir::Graph::deserialize(path);

    if (!graph) {
      throw std::runtime_error(
          "Failed to deserialize XMODEL: " + path);
    }
    for (const auto* child :
         graph->get_root_subgraph()->children_topological_sort()) {
      count_device_subgraphs(
          child, dpu_subgraphs, cpu_subgraphs, first_dpu);
    }
    if (dpu_subgraphs == 0) throw std::runtime_error("No DPU subgraph found");
  }

  std::unique_ptr<xir::Graph> graph;
  int dpu_subgraphs = 0;
  int cpu_subgraphs = 0;
  const xir::Subgraph* first_dpu = nullptr;
};

static int get_fix_point(const xir::Tensor* tensor, int fallback) {
  try {
    if (tensor->has_attr("fix_point")) {
      return tensor->get_attr<int>("fix_point");
    }
  } catch (...) {}

  try {
    if (tensor->has_attr("fixpos")) {
      return tensor->get_attr<int>("fixpos");
    }
  } catch (...) {}

  return fallback;
}

struct StageTimes {
  size_t job = 0;
  size_t dataset_index = 0;
  int lane = -1;
  int pre_worker = -1;
  int post_worker = -1;

  double slot_wait_ms = 0.0;
  double io_ms = 0.0;
  double preprocess_ms = 0.0;
  double pre_dpu_queue_ms = 0.0;
  double input_sync_ms = 0.0;
  double dpu_ms = 0.0;
  double output_sync_ms = 0.0;
  double dpu_post_queue_ms = 0.0;
  double postprocess_ms = 0.0;
  double e2e_ms = 0.0;
  double completion_s = 0.0;
};

class FrameSlot {
 public:
  FrameSlot(
      int lane_id,
      int slot_id,
      const xir::Tensor* input_tensor,
      const xir::Tensor* output_tensor)
      : lane(lane_id),
        id(slot_id),
        input(std::make_unique<OwnedTensorBuffer>(input_tensor)),
        output(std::make_unique<OwnedTensorBuffer>(output_tensor)),
        mask(static_cast<size_t>(kH * kW), 0),
        morphology_scratch(static_cast<size_t>(kH * kW), 0) {
    input_ptrs.push_back(&input->buffer);
    output_ptrs.push_back(&output->buffer);
  }

  int8_t* input_data() {
    return reinterpret_cast<int8_t*>(
        input->storage.data());
  }

  int8_t* output_data() {
    return reinterpret_cast<int8_t*>(
        output->storage.data());
  }

  int lane = 0;
  int id = 0;

  size_t job = 0;
  size_t dataset_index = 0;

  Clock::time_point arrival{};
  Clock::time_point queued_dpu{};
  Clock::time_point queued_post{};

  std::unique_ptr<OwnedTensorBuffer> input;
  std::unique_ptr<OwnedTensorBuffer> output;

  std::vector<vart::TensorBuffer*> input_ptrs;
  std::vector<vart::TensorBuffer*> output_ptrs;

  std::vector<uint8_t> mask;
  std::vector<uint8_t> morphology_scratch;
};

class DpuLane {
 public:
  DpuLane(
      int lane_id,
      const ModelContext& model,
      int slots)
      : id_(lane_id) {
    if (model.cpu_subgraphs == 0 && model.dpu_subgraphs == 1) {
      runner_ = vart::Runner::create_runner(model.first_dpu, "run");
    } else {
      attrs_ = xir::Attrs::create();
      runner_ = vitis::ai::GraphRunner::create_graph_runner(
          model.graph.get(), attrs_.get());
    }

    if (!runner_) {
      throw std::runtime_error("Failed to create VART runner");
    }

    auto inputs = runner_->get_input_tensors();
    auto outputs = runner_->get_output_tensors();

    if (inputs.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error(
          "The model must expose exactly 1 input and 1 output");
    }

    input_tensor_ = inputs[0];
    output_tensor_ = outputs[0];

    validate_tensors();

    input_fix_ = get_fix_point(
        input_tensor_,
        kFallbackInputFix);

    output_fix_ = get_fix_point(
        output_tensor_,
        kFallbackOutputFix);

    input_scale_ =
        std::exp2(static_cast<float>(input_fix_));

    output_scale_ =
        std::exp2(-static_cast<float>(output_fix_));

    slots_.reserve(static_cast<size_t>(slots));

    for (int i = 0; i < slots; ++i) {
      slots_.push_back(
          std::make_unique<FrameSlot>(
              id_,
              i,
              input_tensor_,
              output_tensor_));
    }
  }

  int id() const { return id_; }
  float input_scale() const { return input_scale_; }
  float output_scale() const { return output_scale_; }
  size_t output_element_bytes() const { return output_element_bytes_; }

  size_t input_bytes() const {
    return input_tensor_->get_data_size();
  }

  size_t output_bytes() const {
    return output_tensor_->get_data_size();
  }

  const std::vector<std::unique_ptr<FrameSlot>>& slots() const {
    return slots_;
  }

  void print_metadata() const {
    auto in = input_tensor_->get_shape();
    auto out = output_tensor_->get_shape();

    std::cout
        << "Input  : ["
        << in[0] << ","
        << in[1] << ","
        << in[2] << ","
        << in[3] << "] INT8 fix="
        << input_fix_
        << "\n";

    std::cout
        << "Output : ["
        << out[0] << ","
        << out[1] << ","
        << out[2] << ","
        << out[3] << "] "
        << (output_element_bytes_ == sizeof(float) ? "FLOAT32" : "INT8")
        << (output_element_bytes_ == sizeof(float) ? "" : " fix=")
        << (output_element_bytes_ == sizeof(float) ? "" : std::to_string(output_fix_))
        << "\n";
  }

  void run_full(FrameSlot& slot, StageTimes& t) {
    auto s0 = Clock::now();

    slot.input->buffer.sync_for_write(
        0,
        slot.input->storage.size());

    auto s1 = Clock::now();

    auto job = runner_->execute_async(
        slot.input_ptrs,
        slot.output_ptrs);

    if (job.second != 0) {
      throw std::runtime_error(
          "execute_async returned non-zero status");
    }

    int status = runner_->wait(
        static_cast<int>(job.first),
        -1);

    auto s2 = Clock::now();

    if (status != 0) {
      throw std::runtime_error(
          "VART wait failed status=" +
          std::to_string(status));
    }

    slot.output->buffer.sync_for_read(
        0,
        slot.output->storage.size());

    auto s3 = Clock::now();

    t.input_sync_ms += elapsed_ms(s0, s1);
    t.dpu_ms += elapsed_ms(s1, s2);
    t.output_sync_ms += elapsed_ms(s2, s3);
  }

  double run_model_only(FrameSlot& slot) {
    auto t0 = Clock::now();

    auto job = runner_->execute_async(
        slot.input_ptrs,
        slot.output_ptrs);

    if (job.second != 0) {
      throw std::runtime_error(
          "execute_async returned non-zero status");
    }

    int status = runner_->wait(
        static_cast<int>(job.first),
        -1);

    auto t1 = Clock::now();

    if (status != 0) {
      throw std::runtime_error(
          "VART wait failed status=" +
          std::to_string(status));
    }

    return elapsed_ms(t0, t1);
  }

 private:
  void validate_tensors() {
    const auto in = input_tensor_->get_shape();
    const auto out = output_tensor_->get_shape();

    if (in != std::vector<int32_t>({1, kH, kW, kC})) {
      throw std::runtime_error(
          "Expected input [1,512,512,4]");
    }

    if (out != std::vector<int32_t>({1, kH, kW, 1})) {
      throw std::runtime_error(
          "Expected output [1,512,512,1]");
    }

    if (input_tensor_->get_data_size() !=
        static_cast<size_t>(kH * kW * kC)) {
      throw std::runtime_error(
          "Expected byte-sized INT8 input");
    }

    const size_t output_elements = static_cast<size_t>(kH * kW);
    const size_t output_bytes = output_tensor_->get_data_size();
    if (output_bytes != output_elements &&
        output_bytes != output_elements * sizeof(float)) {
      throw std::runtime_error(
          "Expected INT8 or FLOAT32 output");
    }
    output_element_bytes_ = output_bytes / output_elements;
  }

  int id_ = 0;

  std::unique_ptr<xir::Attrs> attrs_;
  std::unique_ptr<vart::Runner> runner_;

  const xir::Tensor* input_tensor_ = nullptr;
  const xir::Tensor* output_tensor_ = nullptr;

  int input_fix_ = kFallbackInputFix;
  int output_fix_ = kFallbackOutputFix;

  float input_scale_ = 32.0f;
  float output_scale_ = 0.25f;
  size_t output_element_bytes_ = 1;

  std::vector<std::unique_ptr<FrameSlot>> slots_;
};

// ============================================================================
// TIFF / PREPROCESS
// ============================================================================

struct PreprocessWorkspace {
  cv::Mat mag1c;
  cv::Mat red;
  cv::Mat green;
  cv::Mat blue;
};

static void read_tif_float_into(const fs::path& path, cv::Mat& out) {
  cv::Mat img = cv::imread(
      path.string(),
      cv::IMREAD_UNCHANGED);

  if (img.empty()) {
    throw std::runtime_error(
        "Cannot read TIFF: " + path.string());
  }

  if (img.channels() != 1) {
    throw std::runtime_error(
        "Expected one-channel TIFF: " + path.string());
  }

  img.convertTo(out, CV_32F);

  if (out.rows != kH || out.cols != kW) {
    std::ostringstream oss;
    oss
        << "Expected "
        << kW
        << "x"
        << kH
        << ", got "
        << out.cols
        << "x"
        << out.rows
        << " for "
        << path;

    throw std::runtime_error(oss.str());
  }

}

static cv::Mat read_tif_float(const fs::path& path) {
  cv::Mat out;
  read_tif_float_into(path, out);
  return out;
}

static inline float clip02(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 2.0f) return 2.0f;
  return x;
}

static void preprocess_into(
    const fs::path& folder,
    float input_scale,
    int8_t* dst,
    PreprocessWorkspace& ws,
    StageTimes& t) {
  auto io0 = Clock::now();

  read_tif_float_into(folder / "mag1c.tif", ws.mag1c);

  read_tif_float_into(folder / "TOA_AVIRIS_640nm.tif", ws.red);

  read_tif_float_into(folder / "TOA_AVIRIS_550nm.tif", ws.green);

  read_tif_float_into(folder / "TOA_AVIRIS_460nm.tif", ws.blue);

  auto io1 = Clock::now();

  t.io_ms += elapsed_ms(io0, io1);

  auto p0 = Clock::now();

  for (int y = 0; y < kH; ++y) {
    const float* m = ws.mag1c.ptr<float>(y);
    const float* r = ws.red.ptr<float>(y);
    const float* g = ws.green.ptr<float>(y);
    const float* b = ws.blue.ptr<float>(y);

    int8_t* out =
        dst
        + static_cast<size_t>(y)
          * kW
          * kC;

    for (int x = 0; x < kW; ++x) {
      const float values[4] = {
          clip02(m[x] / 1750.0f),
          clip02(r[x] / 60.0f),
          clip02(g[x] / 60.0f),
          clip02(b[x] / 60.0f)
      };

      for (int c = 0; c < 4; ++c) {
        int q = static_cast<int>(
            std::lrint(values[c] * input_scale));

        q = std::max(
            -128,
            std::min(127, q));

        out[x * 4 + c] =
            static_cast<int8_t>(q);
      }
    }
  }

  auto p1 = Clock::now();

  t.preprocess_ms +=
      t.io_ms
      + elapsed_ms(p0, p1);
}

static cv::Mat load_label(const fs::path& folder) {
  cv::Mat f =
      read_tif_float(
          folder / "labelbinary.tif");

  cv::Mat label(
      kH,
      kW,
      CV_8U);

  for (int y = 0; y < kH; ++y) {
    const float* src = f.ptr<float>(y);
    uint8_t* dst = label.ptr<uint8_t>(y);

    for (int x = 0; x < kW; ++x) {
      dst[x] = src[x] > 0.0f ? 1 : 0;
    }
  }

  return label;
}

// ============================================================================
// POSTPROCESS
// ============================================================================

static inline float output_score(
    const int8_t* output,
    size_t element_bytes,
    size_t index) {
  if (element_bytes == sizeof(float)) {
    return reinterpret_cast<const float*>(output)[index];
  }
  return static_cast<float>(output[index]);
}

static inline int8_t quantized_output_score(
    const int8_t* output,
    size_t element_bytes,
    float output_scale,
    size_t index) {
  if (element_bytes == 1) return output[index];
  const int q = static_cast<int>(
      std::lrint(output_score(output, element_bytes, index) / output_scale));
  return static_cast<int8_t>(std::max(-128, std::min(127, q)));
}

static void postprocess_into_mask(
    const int8_t* output,
    size_t element_bytes,
    uint8_t* mask,
    uint8_t* eroded) {
  const auto at = [](const uint8_t* data, int y, int x) {
    return data[static_cast<size_t>(y) * kW + x] != 0;
  };

  for (size_t i = 0;
       i < static_cast<size_t>(kH * kW);
       ++i) {
    mask[i] = output_score(output, element_bytes, i) > 0.0f ? 1 : 0;
  }

  // Same 3x3 cross binary opening used by Testes/Teste_Unet.py.
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      bool keep = at(mask, y, x);
      if (y > 0) keep = keep && at(mask, y - 1, x);
      if (y + 1 < kH) keep = keep && at(mask, y + 1, x);
      if (x > 0) keep = keep && at(mask, y, x - 1);
      if (x + 1 < kW) keep = keep && at(mask, y, x + 1);
      eroded[static_cast<size_t>(y) * kW + x] = keep ? 1 : 0;
    }
  }
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      bool keep = at(eroded, y, x);
      if (y > 0) keep = keep || at(eroded, y - 1, x);
      if (y + 1 < kH) keep = keep || at(eroded, y + 1, x);
      if (x > 0) keep = keep || at(eroded, y, x - 1);
      if (x + 1 < kW) keep = keep || at(eroded, y, x + 1);
      mask[static_cast<size_t>(y) * kW + x] = keep ? 1 : 0;
    }
  }
}

// ============================================================================
// METRICS
// ============================================================================

struct Counts {
  uint64_t tp = 0;
  uint64_t fp = 0;
  uint64_t fn = 0;
  uint64_t tn = 0;
};

struct Metrics {
  double precision = 0.0;
  double recall = 0.0;
  double f1 = 0.0;
  double iou = 0.0;
  double accuracy = 0.0;
};

static Counts confusion(
    const uint8_t* pred,
    const cv::Mat& label) {
  Counts c;

  for (int y = 0; y < kH; ++y) {
    const uint8_t* l =
        label.ptr<uint8_t>(y);

    for (int x = 0; x < kW; ++x) {
      const bool p =
          pred[
              static_cast<size_t>(y)
              * kW
              + x]
          != 0;

      const bool g =
          l[x] != 0;

      if (p && g) ++c.tp;
      else if (p && !g) ++c.fp;
      else if (!p && g) ++c.fn;
      else ++c.tn;
    }
  }

  return c;
}

static Metrics calc_metrics(const Counts& c) {
  Metrics m;

  if (c.tp + c.fp) {
    m.precision =
        static_cast<double>(c.tp)
        /
        static_cast<double>(c.tp + c.fp);
  }

  if (c.tp + c.fn) {
    m.recall =
        static_cast<double>(c.tp)
        /
        static_cast<double>(c.tp + c.fn);
  }

  if (m.precision + m.recall > 0.0) {
    m.f1 =
        2.0
        * m.precision
        * m.recall
        /
        (m.precision + m.recall);
  }

  if (c.tp + c.fp + c.fn) {
    m.iou =
        static_cast<double>(c.tp)
        /
        static_cast<double>(
            c.tp + c.fp + c.fn);
  }

  const uint64_t total =
      c.tp + c.fp + c.fn + c.tn;

  if (total) {
    m.accuracy =
        static_cast<double>(c.tp + c.tn)
        /
        static_cast<double>(total);
  }

  return m;
}

// ============================================================================
// STATS
// ============================================================================

struct Stats {
  size_t count = 0;

  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stddev = 0.0;
  double cv = 0.0;
};

static Stats summarize(std::vector<double> v) {
  Stats s;

  if (v.empty()) return s;

  std::sort(v.begin(), v.end());

  s.count = v.size();
  s.min = v.front();
  s.max = v.back();

  s.mean =
      std::accumulate(
          v.begin(),
          v.end(),
          0.0)
      /
      static_cast<double>(v.size());

  auto q = [&](double p) {
    const double x =
        p
        * static_cast<double>(
            v.size() - 1);

    const size_t i =
        static_cast<size_t>(x);

    const double f =
        x
        - static_cast<double>(i);

    return
        v[i] * (1.0 - f)
        +
        v[
            std::min(
                i + 1,
                v.size() - 1)]
        * f;
  };

  s.median = q(0.50);
  s.p90 = q(0.90);
  s.p95 = q(0.95);
  s.p99 = q(0.99);

  double ss = 0.0;

  for (double x : v) {
    const double d = x - s.mean;
    ss += d * d;
  }

  s.stddev =
      std::sqrt(
          ss
          /
          static_cast<double>(v.size()));

  s.cv =
      s.mean != 0.0
      ? s.stddev / s.mean
      : 0.0;

  return s;
}

struct BenchmarkResult {
  std::string mode;

  int batch = 1;
  int runners = 1;
  int pre_workers = 0;
  int post_workers = 0;
  int slots_per_runner = 1;

  size_t completed = 0;

  double wall_s = 0.0;
  double throughput_fps = 0.0;

  Stats latency;
  Stats dpu;
  Stats preprocess;
  Stats postprocess;
  Stats inter_completion;

  std::vector<StageTimes> samples;
  std::vector<PowerRail> power_rails;
};

// ============================================================================
// BOUNDED QUEUE / START GATE
// ============================================================================

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity)
      : buf_(capacity),
        capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::runtime_error(
          "Queue capacity cannot be zero");
    }
  }

  void push(T v) {
    std::unique_lock<std::mutex> lock(mu_);

    not_full_.wait(
        lock,
        [&] {
          return count_ < capacity_;
        });

    buf_[tail_] = std::move(v);

    tail_ =
        (tail_ + 1)
        % capacity_;

    ++count_;

    lock.unlock();
    not_empty_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mu_);

    not_empty_.wait(
        lock,
        [&] {
          return count_ > 0;
        });

    T v =
        std::move(buf_[head_]);

    head_ =
        (head_ + 1)
        % capacity_;

    --count_;

    lock.unlock();
    not_full_.notify_one();

    return v;
  }

 private:
  std::vector<T> buf_;

  size_t capacity_ = 0;
  size_t head_ = 0;
  size_t tail_ = 0;
  size_t count_ = 0;

  std::mutex mu_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

class StartGate {
 public:
  explicit StartGate(int expected)
      : expected_(expected) {}

  void worker_ready_and_wait() {
    std::unique_lock<std::mutex> lock(mu_);

    ++ready_;
    cv_.notify_all();

    cv_.wait(
        lock,
        [&] {
          return released_;
        });
  }

  void wait_ready() {
    std::unique_lock<std::mutex> lock(mu_);

    cv_.wait(
        lock,
        [&] {
          return ready_ == expected_;
        });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mu_);

    released_ = true;
    cv_.notify_all();
  }

 private:
  int expected_ = 0;
  int ready_ = 0;

  bool released_ = false;

  std::mutex mu_;
  std::condition_variable cv_;
};

// ============================================================================
// CREATE LANES
// ============================================================================

static std::vector<std::unique_ptr<DpuLane>>
make_lanes(
    const ModelContext& model,
    int runners,
    int slots_per_runner) {
  std::vector<std::unique_ptr<DpuLane>> lanes;

  lanes.reserve(
      static_cast<size_t>(runners));

  for (int i = 0; i < runners; ++i) {
    lanes.push_back(
        std::make_unique<DpuLane>(
            i,
            model,
            slots_per_runner));
  }

  return lanes;
}

// ============================================================================
// PREPARE INPUTS
// ============================================================================

static std::vector<std::vector<int8_t>>
prepare_inputs(
    const std::vector<DatasetItem>& dataset,
    float input_scale,
    size_t max_items) {
  std::vector<std::vector<int8_t>> cache;

  const size_t count = std::min(dataset.size(), max_items);
  cache.reserve(count);

  PreprocessWorkspace ws;

  for (size_t i = 0; i < count; ++i) {
    std::vector<int8_t> input(
        static_cast<size_t>(kH * kW * kC));

    StageTimes t;

    preprocess_into(
        dataset[i].folder,
        input_scale,
        input.data(),
        ws,
        t);

    cache.push_back(
        std::move(input));
  }

  return cache;
}

// ============================================================================
// VALIDATION
// ============================================================================

static void add_counts(Counts& dst, const Counts& src) {
  dst.tp += src.tp;
  dst.fp += src.fp;
  dst.fn += src.fn;
  dst.tn += src.tn;
}

static double average_precision_from_int8_scores(
    const std::array<uint64_t, 256>& positives,
    const std::array<uint64_t, 256>& negatives) {
  const uint64_t total_positive =
      std::accumulate(positives.begin(), positives.end(), uint64_t{0});
  if (total_positive == 0) return 0.0;

  uint64_t tp = 0;
  uint64_t fp = 0;
  double ap = 0.0;
  for (int bin = 255; bin >= 0; --bin) {
    const uint64_t new_tp = positives[static_cast<size_t>(bin)];
    tp += new_tp;
    fp += negatives[static_cast<size_t>(bin)];
    if (new_tp != 0) {
      const double precision = static_cast<double>(tp) / static_cast<double>(tp + fp);
      ap += precision * static_cast<double>(new_tp) /
            static_cast<double>(total_positive);
    }
  }
  return ap;
}

static void run_self_test() {
  std::vector<int8_t> logits(static_cast<size_t>(kH * kW), -1);
  std::vector<uint8_t> mask(static_cast<size_t>(kH * kW), 0);
  std::vector<uint8_t> scratch(static_cast<size_t>(kH * kW), 0);
  const int cy = kH / 2;
  const int cx = kW / 2;
  const auto set_positive = [&](int y, int x) {
    logits[static_cast<size_t>(y) * kW + x] = 1;
  };
  set_positive(cy, cx);
  set_positive(cy - 1, cx);
  set_positive(cy + 1, cx);
  set_positive(cy, cx - 1);
  set_positive(cy, cx + 1);
  postprocess_into_mask(logits.data(), 1, mask.data(), scratch.data());
  const uint64_t retained =
      std::accumulate(mask.begin(), mask.end(), uint64_t{0});
  if (retained != 5) throw std::runtime_error("morphology self-test failed");

  std::vector<float> float_logits(static_cast<size_t>(kH * kW), -0.25f);
  float_logits[static_cast<size_t>(cy) * kW + cx] = 0.25f;
  postprocess_into_mask(
      reinterpret_cast<const int8_t*>(float_logits.data()),
      sizeof(float),
      mask.data(),
      scratch.data());
  if (mask[static_cast<size_t>(cy) * kW + cx] != 0) {
    throw std::runtime_error("FLOAT32 morphology self-test failed");
  }
  if (quantized_output_score(
          reinterpret_cast<const int8_t*>(float_logits.data()),
          sizeof(float), 0.25f,
          static_cast<size_t>(cy) * kW + cx) != 1) {
    throw std::runtime_error("FLOAT32 score self-test failed");
  }

  std::array<uint64_t, 256> positive{};
  std::array<uint64_t, 256> negative{};
  positive[130] = 1;
  negative[129] = 1;
  positive[128] = 1;
  const double ap = average_precision_from_int8_scores(positive, negative);
  if (std::abs(ap - (5.0 / 6.0)) > 1e-12) {
    throw std::runtime_error("AUPRC self-test failed");
  }
  std::cout << "Self-test OK\n";
}

static void run_validation(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          model,
          1,
          1);

  auto& lane =
      *lanes.front();

  auto& slot =
      *lane.slots().front();

  lane.print_metadata();

  Counts global;
  Counts strong;
  Counts weak;
  Counts no_plume;
  std::array<uint64_t, 256> positive_scores{};
  std::array<uint64_t, 256> negative_scores{};

  std::ofstream per_image(
      fs::path(o.out)
      / "metricas_por_imagem.csv");

  per_image
      << "model,dataset,id,has_plume,qplume,difficulty,TP,FP,FN,TN,"
      << "precision,recall,f1,iou,accuracy\n";

  PreprocessWorkspace ws;

  for (size_t i = 0; i < dataset.size(); ++i) {
    const auto& item = dataset[i];

    const fs::path& folder = item.folder;

    StageTimes t;

    preprocess_into(
        folder,
        lane.input_scale(),
        slot.input_data(),
        ws,
        t);

    lane.run_full(
        slot,
        t);

    postprocess_into_mask(
        slot.output_data(),
        lane.output_element_bytes(),
        slot.mask.data(),
        slot.morphology_scratch.data());

    cv::Mat label =
        load_label(folder);

    Counts c =
        confusion(
            slot.mask.data(),
            label);

    Metrics m =
        calc_metrics(c);

    add_counts(global, c);

    const uint64_t positive_pixels = c.tp + c.fn;
    std::string difficulty = "no_plume";
    if (positive_pixels == 0) {
      add_counts(no_plume, c);
    } else if (item.qplume >= 1000.0 || positive_pixels > 1000) {
      difficulty = "strong";
      add_counts(strong, c);
    } else {
      difficulty = "weak";
      add_counts(weak, c);
    }

    const int8_t* scores = slot.output_data();
    for (int y = 0; y < kH; ++y) {
      const uint8_t* truth = label.ptr<uint8_t>(y);
      for (int x = 0; x < kW; ++x) {
        const size_t pixel = static_cast<size_t>(y) * kW + x;
        const int8_t score = quantized_output_score(
            scores,
            lane.output_element_bytes(),
            lane.output_scale(),
            pixel);
        const size_t bin = static_cast<size_t>(static_cast<int>(score) + 128);
        (truth[x] != 0 ? positive_scores : negative_scores)[bin]++;
      }
    }

    per_image
        << fs::path(o.model).filename().string() << ","
        << '"' << o.dataset << "\","
        << item.id << ","
        << item.has_plume << ","
        << item.qplume << ","
        << difficulty << ","
        << c.tp << ","
        << c.fp << ","
        << c.fn << ","
        << c.tn << ","
        << std::setprecision(12)
        << m.precision << ","
        << m.recall << ","
        << m.f1 << ","
        << m.iou << ","
        << m.accuracy
        << "\n";

    std::cout
        << "[VAL "
        << (i + 1)
        << "/"
        << dataset.size()
        << "] "
        << item.id
        << " F1="
        << std::fixed
        << std::setprecision(4)
        << m.f1
        << " IoU="
        << m.iou
        << "\n";
  }

  Metrics gm =
      calc_metrics(global);
  const Metrics strong_metrics = calc_metrics(strong);
  const Metrics weak_metrics = calc_metrics(weak);
  const double auprc =
      average_precision_from_int8_scores(positive_scores, negative_scores);
  const double fpr_no_plume =
      no_plume.fp + no_plume.tn == 0
      ? 0.0
      : static_cast<double>(no_plume.fp) /
        static_cast<double>(no_plume.fp + no_plume.tn);

  std::ofstream g(
      fs::path(o.out)
      / "metricas_globais.csv");

  g
      << "model,dataset,num_imagens,cpu_cores,runners,pre_workers,post_workers,"
      << "slots_per_runner,pin,TP,FP,FN,TN,"
      << "precision_global,recall_global,"
      << "f1_global,f1_strong,f1_weak,iou_global,auprc,"
      << "fpr_no_plume,accuracy_global\n";

  g
      << fs::path(o.model).filename().string() << ","
      << '"' << o.dataset << "\","
      << dataset.size() << ","
      << o.cpu_cores << ","
      << o.runners << ","
      << o.pre_workers << ","
      << o.post_workers << ","
      << o.slots_per_runner << ","
      << (o.pin ? "true" : "false") << ","
      << global.tp << ","
      << global.fp << ","
      << global.fn << ","
      << global.tn << ","
      << std::setprecision(12)
      << gm.precision << ","
      << gm.recall << ","
      << gm.f1 << ","
      << strong_metrics.f1 << ","
      << weak_metrics.f1 << ","
      << gm.iou << ","
      << auprc << ","
      << fpr_no_plume << ","
      << gm.accuracy
      << "\n";

  std::cout
      << "\nVALIDATION GLOBAL\n"
      << "Precision="
      << gm.precision
      << " Recall="
      << gm.recall
      << " F1="
      << gm.f1
      << " F1-Strong="
      << strong_metrics.f1
      << " F1-Weak="
      << weak_metrics.f1
      << " IoU="
      << gm.iou
      << " AUPRC="
      << auprc
      << " FPR-No-Plume="
      << fpr_no_plume
      << " Accuracy="
      << gm.accuracy
      << "\n";
}

// ============================================================================
// BUILD RESULT STATS
// ============================================================================

static BenchmarkResult finalize_result(
    BenchmarkResult r) {
  std::vector<double> lat;
  std::vector<double> dpu;
  std::vector<double> prep;
  std::vector<double> post;
  std::vector<double> completion;

  for (const auto& t : r.samples) {
    lat.push_back(t.e2e_ms);

    if (t.dpu_ms > 0.0) {
      dpu.push_back(t.dpu_ms);
    }

    if (t.preprocess_ms > 0.0) {
      prep.push_back(t.preprocess_ms);
    }

    if (t.postprocess_ms > 0.0) {
      post.push_back(t.postprocess_ms);
    }

    completion.push_back(t.completion_s);
  }

  r.latency = summarize(lat);
  r.dpu = summarize(dpu);
  r.preprocess = summarize(prep);
  r.postprocess = summarize(post);

  std::sort(
      completion.begin(),
      completion.end());

  std::vector<double> intervals;

  if (completion.size() > 1) {
    intervals.reserve(
        completion.size() - 1);

    for (size_t i = 1;
         i < completion.size();
         ++i) {
      intervals.push_back(
          (completion[i] - completion[i - 1])
          * 1000.0);
    }
  }

  r.inter_completion =
      summarize(intervals);

  return r;
}

// ============================================================================
// BASELINE BATCH1
// ============================================================================

static std::vector<BenchmarkResult>
run_baseline(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          model,
          1,
          1);

  auto& lane =
      *lanes.front();

  auto& slot =
      *lane.slots().front();

  auto cache =
      prepare_inputs(
          dataset,
          lane.input_scale(),
          1);

  // ------------------------------------------------------------------------
  // baseline model-only:
  // only execute_async + wait timed
  // ------------------------------------------------------------------------

  // Resident prepared input. Model-only timing contains no preprocessing,
  // memcpy, quantization or buffer sync.
  const auto& baseline_src = cache.front();

  std::memcpy(
      slot.input_data(),
      baseline_src.data(),
      baseline_src.size());

  slot.input->buffer.sync_for_write(
      0,
      slot.input->storage.size());

  for (int i = 0; i < o.warmup; ++i) {
    (void)lane.run_model_only(slot);
  }

  BenchmarkResult mo;
  mo.mode = "baseline_model_only";
  mo.batch = 1;
  mo.runners = 1;
  mo.slots_per_runner = 1;
  mo.samples.resize(
      static_cast<size_t>(
          o.baseline_repeats));

  auto wall0 = Clock::now();
  PowerMonitor mo_power(o);
  mo_power.start();

  for (int i = 0;
       i < o.baseline_repeats;
       ++i) {
    StageTimes& t =
        mo.samples[
            static_cast<size_t>(i)];

    t.job = static_cast<size_t>(i);
    t.dataset_index =
        static_cast<size_t>(i)
        % dataset.size();

    const double dpu_ms =
        lane.run_model_only(slot);

    t.dpu_ms = dpu_ms;
    t.e2e_ms = dpu_ms;

    t.completion_s =
        std::chrono::duration<double>(
            Clock::now() - wall0)
        .count();
  }

  auto wall1 = Clock::now();
  mo_power.stop();
  mo.power_rails = mo_power.take();

  mo.wall_s =
      std::chrono::duration<double>(
          wall1 - wall0)
      .count();

  mo.completed =
      mo.samples.size();

  mo.throughput_fps =
      static_cast<double>(mo.completed)
      /
      mo.wall_s;

  mo = finalize_result(
      std::move(mo));

  // ------------------------------------------------------------------------
  // baseline end-to-end:
  // TIFF -> preprocess -> DPU -> dequant/sigmoid/threshold -> mask
  // ------------------------------------------------------------------------

  PreprocessWorkspace ws;

  for (int pass = 0;
       pass < 1;
       ++pass) {
    for (size_t i = 0;
         i < dataset.size();
         ++i) {
      StageTimes t;

      preprocess_into(
          dataset[i].folder,
          lane.input_scale(),
          slot.input_data(),
          ws,
          t);

      lane.run_full(
          slot,
          t);

      postprocess_into_mask(
          slot.output_data(),
          lane.output_element_bytes(),
          slot.mask.data(),
          slot.morphology_scratch.data());
    }
  }

  BenchmarkResult e2e;
  e2e.mode = "baseline_end_to_end";
  e2e.batch = 1;
  e2e.runners = 1;
  e2e.pre_workers = 1;
  e2e.post_workers = 1;
  e2e.slots_per_runner = 1;

  const size_t total =
      static_cast<size_t>(
          o.baseline_e2e_passes)
      * dataset.size();

  e2e.samples.resize(total);

  auto e0 = Clock::now();
  PowerMonitor e2e_power(o);
  e2e_power.start();

  size_t job_index = 0;

  for (int pass = 0;
       pass < o.baseline_e2e_passes;
       ++pass) {
    for (size_t di = 0;
         di < dataset.size();
         ++di) {
      StageTimes& t =
          e2e.samples[job_index];

      t.job = job_index;
      t.dataset_index = di;

      auto t0 = Clock::now();

      preprocess_into(
          dataset[di].folder,
          lane.input_scale(),
          slot.input_data(),
          ws,
          t);

      lane.run_full(
          slot,
          t);

      auto p0 = Clock::now();

      postprocess_into_mask(
          slot.output_data(),
          lane.output_element_bytes(),
          slot.mask.data(),
          slot.morphology_scratch.data());

      auto p1 = Clock::now();

      t.postprocess_ms =
          elapsed_ms(p0, p1);

      t.e2e_ms =
          elapsed_ms(t0, p1);

      t.completion_s =
          std::chrono::duration<double>(
              p1 - e0)
          .count();

      ++job_index;
    }
  }

  auto e1 = Clock::now();
  e2e_power.stop();
  e2e.power_rails = e2e_power.take();

  e2e.wall_s =
      std::chrono::duration<double>(
          e1 - e0)
      .count();

  e2e.completed =
      e2e.samples.size();

  e2e.throughput_fps =
      static_cast<double>(e2e.completed)
      /
      e2e.wall_s;

  e2e = finalize_result(
      std::move(e2e));

  return {
      std::move(mo),
      std::move(e2e)
  };
}

// ============================================================================
// MAX MODEL-ONLY THROUGHPUT
// ============================================================================

static BenchmarkResult
run_max_model_only(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          model,
          o.runners,
          1);

  lanes.front()->print_metadata();

  auto cache =
      prepare_inputs(
          dataset,
          lanes.front()->input_scale(),
          static_cast<size_t>(o.runners));

  // Pre-fill one resident input per runner.
  for (int r = 0; r < o.runners; ++r) {
    auto& lane =
        *lanes[
            static_cast<size_t>(r)];

    auto& slot =
        *lane.slots().front();

    const auto& src =
        cache[
            static_cast<size_t>(r)
            % cache.size()];

    std::memcpy(
        slot.input_data(),
        src.data(),
        src.size());

    // unchanged resident input: sync once before benchmark
    slot.input->buffer.sync_for_write(
        0,
        slot.input->storage.size());

    for (int i = 0;
         i < o.warmup;
         ++i) {
      (void)lane.run_model_only(slot);
    }
  }

  BenchmarkResult result;
  result.mode = "max_model_only_throughput";
  result.batch = 1;
  result.runners = o.runners;
  result.slots_per_runner = 1;
  result.samples.resize(
      static_cast<size_t>(
          o.iterations));

  std::atomic<size_t> next{0};

  StartGate gate(
      o.runners);

  Clock::time_point run_start{};

  std::vector<std::thread> threads;
  threads.reserve(
      static_cast<size_t>(
          o.runners));

  for (int r = 0;
       r < o.runners;
       ++r) {
    threads.emplace_back(
        [&, r] {
          if (o.pin) {
            pin_current_thread(r);
          }

          auto& lane =
              *lanes[
                  static_cast<size_t>(r)];

          auto& slot =
              *lane.slots().front();

          gate.worker_ready_and_wait();

          for (;;) {
            const size_t j =
                next.fetch_add(1);

            if (j >=
                static_cast<size_t>(
                    o.iterations)) {
              break;
            }

            StageTimes& t =
                result.samples[j];

            t.job = j;
            t.dataset_index =
                j % dataset.size();
            t.lane = r;

            const double dpu_ms =
                lane.run_model_only(slot);

            t.dpu_ms = dpu_ms;
            t.e2e_ms = dpu_ms;

            t.completion_s =
                std::chrono::duration<double>(
                    Clock::now()
                    - run_start)
                .count();
          }
        });
  }

  gate.wait_ready();

  run_start =
      Clock::now();

  PowerMonitor power(o);
  power.start();

  gate.release();

  for (auto& t : threads) {
    t.join();
  }

  auto run_end =
      Clock::now();

  power.stop();
  result.power_rails = power.take();

  result.wall_s =
      std::chrono::duration<double>(
          run_end - run_start)
      .count();

  result.completed =
      result.samples.size();

  result.throughput_fps =
      static_cast<double>(
          result.completed)
      /
      result.wall_s;

  return finalize_result(
      std::move(result));
}

// ============================================================================
// MAX END-TO-END PIPELINE
// ============================================================================

static BenchmarkResult
run_max_e2e(
    const Options& o,
    const std::vector<DatasetItem>& dataset,
    ModelContext& model) {
  auto lanes =
      make_lanes(
          model,
          o.runners,
          o.slots_per_runner);

  lanes.front()->print_metadata();

  // Warm each lane with first sample.
  for (int r = 0;
       r < o.runners;
       ++r) {
    auto& lane =
        *lanes[
            static_cast<size_t>(r)];

    auto& slot =
        *lane.slots().front();

    PreprocessWorkspace ws;
    StageTimes t;

    preprocess_into(
        dataset.front().folder,
        lane.input_scale(),
        slot.input_data(),
        ws,
        t);

    for (int i = 0;
         i < o.warmup;
         ++i) {
      StageTimes w;
      lane.run_full(
          slot,
          w);
    }
  }

  BenchmarkResult result;
  result.mode = "max_end_to_end_throughput";
  result.batch = 1;
  result.runners = o.runners;
  result.pre_workers = o.pre_workers;
  result.post_workers = o.post_workers;
  result.slots_per_runner =
      o.slots_per_runner;
  result.samples.resize(
      static_cast<size_t>(
          o.iterations));

  const size_t total_slots =
      static_cast<size_t>(
          o.runners
          * o.slots_per_runner);

  BoundedQueue<FrameSlot*> free_q(
      total_slots);

  std::vector<
      std::unique_ptr<
          BoundedQueue<FrameSlot*>>>
      dpu_q;

  dpu_q.reserve(
      static_cast<size_t>(
          o.runners));

  for (int r = 0;
       r < o.runners;
       ++r) {
    dpu_q.push_back(
        std::make_unique<
            BoundedQueue<FrameSlot*>>(
                static_cast<size_t>(
                    o.slots_per_runner)));
  }

  BoundedQueue<FrameSlot*> post_q(
      total_slots);

  // Interleave lane slots.
  for (int s = 0;
       s < o.slots_per_runner;
       ++s) {
    for (int r = 0;
         r < o.runners;
         ++r) {
      free_q.push(
          lanes[
              static_cast<size_t>(r)]
          ->slots()[
              static_cast<size_t>(s)]
          .get());
    }
  }

  const int participants =
      o.pre_workers
      + o.runners
      + o.post_workers;

  StartGate gate(
      participants);

  std::atomic<size_t> next_job{0};

  Clock::time_point run_start{};

  std::vector<std::thread> pre_threads;
  std::vector<std::thread> dpu_threads;
  std::vector<std::thread> post_threads;

  // ------------------------------------------------------------------------
  // PRE
  // ------------------------------------------------------------------------

  for (int pw = 0;
       pw < o.pre_workers;
       ++pw) {
    pre_threads.emplace_back(
        [&, pw] {
          if (o.pin) {
            pin_current_thread(pw);
          }

          PreprocessWorkspace ws;

          gate.worker_ready_and_wait();

          for (;;) {
            const size_t j =
                next_job.fetch_add(1);

            if (j >=
                static_cast<size_t>(
                    o.iterations)) {
              break;
            }

            const size_t di =
                j % dataset.size();

            auto arrival =
                Clock::now();

            FrameSlot* slot =
                free_q.pop();

            auto got_slot =
                Clock::now();

            StageTimes& t =
                result.samples[j];

            t.job = j;
            t.dataset_index = di;
            t.lane = slot->lane;
            t.pre_worker = pw;
            t.slot_wait_ms =
                elapsed_ms(
                    arrival,
                    got_slot);

            slot->job = j;
            slot->dataset_index = di;
            slot->arrival = arrival;

            preprocess_into(
                dataset[di].folder,
                lanes[
                    static_cast<size_t>(
                        slot->lane)]
                ->input_scale(),
                slot->input_data(),
                ws,
                t);

            slot->queued_dpu =
                Clock::now();

            dpu_q[
                static_cast<size_t>(
                    slot->lane)]
            ->push(slot);
          }
        });
  }

  // ------------------------------------------------------------------------
  // DPU
  // ------------------------------------------------------------------------

  for (int r = 0;
       r < o.runners;
       ++r) {
    dpu_threads.emplace_back(
        [&, r] {
          if (o.pin) {
            pin_current_thread(
                o.pre_workers
                + o.post_workers
                + r);
          }

          auto& lane =
              *lanes[
                  static_cast<size_t>(r)];

          gate.worker_ready_and_wait();

          for (;;) {
            FrameSlot* slot =
                dpu_q[
                    static_cast<size_t>(r)]
                ->pop();

            if (slot == nullptr) {
              break;
            }

            StageTimes& t =
                result.samples[
                    slot->job];

            t.pre_dpu_queue_ms =
                elapsed_ms(
                    slot->queued_dpu,
                    Clock::now());

            lane.run_full(
                *slot,
                t);

            slot->queued_post =
                Clock::now();

            post_q.push(slot);
          }
        });
  }

  // ------------------------------------------------------------------------
  // POST
  // ------------------------------------------------------------------------

  for (int pw = 0;
       pw < o.post_workers;
       ++pw) {
    post_threads.emplace_back(
        [&, pw] {
          if (o.pin) {
            pin_current_thread(
                o.pre_workers
                + pw);
          }

          gate.worker_ready_and_wait();

          for (;;) {
            FrameSlot* slot =
                post_q.pop();

            if (slot == nullptr) {
              break;
            }

            StageTimes& t =
                result.samples[
                    slot->job];

            t.post_worker = pw;

            t.dpu_post_queue_ms =
                elapsed_ms(
                    slot->queued_post,
                    Clock::now());

            auto p0 =
                Clock::now();

            postprocess_into_mask(
                slot->output_data(),
                lanes[
                    static_cast<size_t>(
                        slot->lane)]
                ->output_element_bytes(),
                slot->mask.data(),
                slot->morphology_scratch.data());

            auto p1 =
                Clock::now();

            t.postprocess_ms =
                elapsed_ms(p0, p1);

            t.e2e_ms =
                elapsed_ms(
                    slot->arrival,
                    p1);

            t.completion_s =
                std::chrono::duration<double>(
                    p1 - run_start)
                .count();

            // return reusable aligned slot
            free_q.push(slot);
          }
        });
  }

  gate.wait_ready();

  run_start =
      Clock::now();

  PowerMonitor power(o);
  power.start();

  gate.release();

  for (auto& t : pre_threads) {
    t.join();
  }

  // stop DPU workers
  for (int r = 0;
       r < o.runners;
       ++r) {
    dpu_q[
        static_cast<size_t>(r)]
    ->push(nullptr);
  }

  for (auto& t : dpu_threads) {
    t.join();
  }

  // stop post workers
  for (int p = 0;
       p < o.post_workers;
       ++p) {
    post_q.push(nullptr);
  }

  for (auto& t : post_threads) {
    t.join();
  }

  auto run_end =
      Clock::now();

  power.stop();
  result.power_rails = power.take();

  result.wall_s =
      std::chrono::duration<double>(
          run_end - run_start)
      .count();

  result.completed =
      result.samples.size();

  result.throughput_fps =
      static_cast<double>(
          result.completed)
      /
      result.wall_s;

  return finalize_result(
      std::move(result));
}

// ============================================================================
// WRITE RESULTS
// ============================================================================

static void write_samples_csv(
    const fs::path& path,
    const BenchmarkResult& r) {
  std::ofstream f(
      path,
      std::ios::app);

  if (f.tellp() == 0) {
    f
        << "mode,job,dataset_index,lane,pre_worker,post_worker,"
        << "slot_wait_ms,io_ms,preprocess_ms,pre_dpu_queue_ms,"
        << "input_sync_ms,dpu_ms,output_sync_ms,dpu_post_queue_ms,"
        << "postprocess_ms,e2e_ms,completion_s,instantaneous_fps\n";
  }

  f << std::setprecision(12);

  for (const auto& t : r.samples) {
    f
        << r.mode << ","
        << t.job << ","
        << t.dataset_index << ","
        << t.lane << ","
        << t.pre_worker << ","
        << t.post_worker << ","
        << t.slot_wait_ms << ","
        << t.io_ms << ","
        << t.preprocess_ms << ","
        << t.pre_dpu_queue_ms << ","
        << t.input_sync_ms << ","
        << t.dpu_ms << ","
        << t.output_sync_ms << ","
        << t.dpu_post_queue_ms << ","
        << t.postprocess_ms << ","
        << t.e2e_ms << ","
        << t.completion_s << ","
        << (
            t.e2e_ms > 0.0
            ? 1000.0 / t.e2e_ms
            : 0.0)
        << "\n";
  }
}

static void write_summary_header(
    std::ofstream& f) {
  f
      << "model,dataset,csv,samples,cpu_cores,pin,"
      << "mode,batch,runners,pre_workers,post_workers,slots_per_runner,"
      << "completed,wall_s,throughput_fps,"
      << "equiv_fps_avg,equiv_fps_min,equiv_fps_max,equiv_fps_p95,equiv_fps_p99,"
      << "completion_fps_avg,completion_fps_min,completion_fps_max,completion_fps_p95,completion_fps_p99,"
      << "latency_mean_ms,latency_median_ms,latency_min_ms,latency_max_ms,"
      << "latency_p90_ms,latency_p95_ms,latency_p99_ms,"
      << "latency_stddev_ms,latency_cv,latency_p99_minus_p50_ms,"
      << "dpu_mean_ms,dpu_p95_ms,dpu_p99_ms,"
      << "preprocess_mean_ms,postprocess_mean_ms,"
      << "inter_completion_mean_ms,inter_completion_p95_ms,inter_completion_p99_ms,"
      << "power_sensor_count,power_samples,power_mean_w,power_min_w,power_max_w,energy_j\n";
}

static std::array<double, 5> power_totals(const BenchmarkResult& r) {
  size_t samples = 0;
  for (const auto& rail : r.power_rails) samples = std::max(samples, rail.watts.size());
  if (samples == 0) return {0, 0, 0, 0, 0};
  std::vector<double> totals(samples, 0.0);
  for (const auto& rail : r.power_rails) {
    for (size_t i = 0; i < rail.watts.size(); ++i) totals[i] += rail.watts[i];
  }
  const Stats stats = summarize(totals);
  return {static_cast<double>(samples), stats.mean, stats.min, stats.max,
          stats.mean * r.wall_s};
}

static void write_summary_row(
    std::ofstream& f,
    const Options& o,
    const BenchmarkResult& r) {
  const auto power = power_totals(r);
  f
      << std::setprecision(12)
      << fs::path(o.model).filename().string() << ","
      << o.dataset << ","
      << o.csv << ","
      << o.samples << ","
      << o.cpu_cores << ","
      << (o.pin ? "true" : "false") << ","
      << r.mode << ","
      << r.batch << ","
      << r.runners << ","
      << r.pre_workers << ","
      << r.post_workers << ","
      << r.slots_per_runner << ","
      << r.completed << ","
      << r.wall_s << ","
      << r.throughput_fps << ","
      << (r.latency.mean > 0.0 ? 1000.0 / r.latency.mean : 0.0) << ","
      << (r.latency.max > 0.0 ? 1000.0 / r.latency.max : 0.0) << ","
      << (r.latency.min > 0.0 ? 1000.0 / r.latency.min : 0.0) << ","
      << (r.latency.p95 > 0.0 ? 1000.0 / r.latency.p95 : 0.0) << ","
      << (r.latency.p99 > 0.0 ? 1000.0 / r.latency.p99 : 0.0) << ","
      << r.throughput_fps << ","
      << (r.inter_completion.max > 0.0 ? 1000.0 / r.inter_completion.max : r.throughput_fps) << ","
      << (r.inter_completion.min > 0.0 ? 1000.0 / r.inter_completion.min : r.throughput_fps) << ","
      << (r.inter_completion.p95 > 0.0 ? 1000.0 / r.inter_completion.p95 : r.throughput_fps) << ","
      << (r.inter_completion.p99 > 0.0 ? 1000.0 / r.inter_completion.p99 : r.throughput_fps) << ","
      << r.latency.mean << ","
      << r.latency.median << ","
      << r.latency.min << ","
      << r.latency.max << ","
      << r.latency.p90 << ","
      << r.latency.p95 << ","
      << r.latency.p99 << ","
      << r.latency.stddev << ","
      << r.latency.cv << ","
      << (r.latency.p99 - r.latency.median) << ","
      << r.dpu.mean << ","
      << r.dpu.p95 << ","
      << r.dpu.p99 << ","
      << r.preprocess.mean << ","
      << r.postprocess.mean << ","
      << r.inter_completion.mean << ","
      << r.inter_completion.p95 << ","
      << r.inter_completion.p99 << ","
      << r.power_rails.size() << ","
      << power[0] << ","
      << power[1] << ","
      << power[2] << ","
      << power[3] << ","
      << power[4]
      << "\n";
}

static void write_power_csv(const fs::path& path, const BenchmarkResult& r) {
  std::ofstream f(path, std::ios::app);
  if (f.tellp() == 0) f << "mode,sensor,path,samples,power_mean_w,power_min_w,power_max_w,energy_j\n";
  for (const auto& rail : r.power_rails) {
    const Stats s = summarize(rail.watts);
    f << std::setprecision(12) << r.mode << ",\"" << rail.name << "\",\""
      << rail.input.string() << "\"," << s.count << ',' << s.mean << ',' << s.min
      << ',' << s.max << ',' << s.mean * r.wall_s << '\n';
  }
}

static void print_result(
    const BenchmarkResult& r) {
  std::cout
      << "\n============================================================\n"
      << r.mode
      << "\n============================================================\n";

  std::cout
      << "batch              = "
      << r.batch
      << "\n"
      << "runners            = "
      << r.runners
      << "\n"
      << "pre_workers        = "
      << r.pre_workers
      << "\n"
      << "post_workers       = "
      << r.post_workers
      << "\n"
      << "slots_per_runner   = "
      << r.slots_per_runner
      << "\n"
      << "completed          = "
      << r.completed
      << "\n"
      << "wall_s             = "
      << std::fixed
      << std::setprecision(6)
      << r.wall_s
      << "\n"
      << "THROUGHPUT FPS     = "
      << r.throughput_fps
      << "\n";

  const auto power = power_totals(r);
  if (power[0] > 0.0) {
    std::cout << "POWER sensors/samples = " << r.power_rails.size() << "/" << power[0]
              << "\nPOWER mean/min/max W = " << power[1] << "/" << power[2] << "/" << power[3]
              << "\nENERGY estimated J   = " << power[4] << "\n";
  }

  std::cout
      << "\nFPS EQUIVALENT FROM PER-JOB LATENCY\n"
      << " average = "
      << (r.latency.mean > 0.0 ? 1000.0 / r.latency.mean : 0.0)
      << "\n"
      << " min     = "
      << (r.latency.max > 0.0 ? 1000.0 / r.latency.max : 0.0)
      << "\n"
      << " max     = "
      << (r.latency.min > 0.0 ? 1000.0 / r.latency.min : 0.0)
      << "\n"
      << " p95     = "
      << (r.latency.p95 > 0.0 ? 1000.0 / r.latency.p95 : 0.0)
      << "\n"
      << " p99     = "
      << (r.latency.p99 > 0.0 ? 1000.0 / r.latency.p99 : 0.0)
      << "\n";

  std::cout
      << "\nLATENCY ms\n"
      << " mean   = "
      << r.latency.mean
      << "\n"
      << " median = "
      << r.latency.median
      << "\n"
      << " min    = "
      << r.latency.min
      << "\n"
      << " max    = "
      << r.latency.max
      << "\n"
      << " p90    = "
      << r.latency.p90
      << "\n"
      << " p95    = "
      << r.latency.p95
      << "\n"
      << " p99    = "
      << r.latency.p99
      << "\n"
      << " stddev = "
      << r.latency.stddev
      << "\n"
      << " CV     = "
      << r.latency.cv
      << "\n"
      << " jitter p99-p50 = "
      << (
          r.latency.p99
          - r.latency.median)
      << " ms\n";

  if (r.dpu.count) {
    std::cout
        << "\nDPU STAGE ms\n"
        << " mean = "
        << r.dpu.mean
        << "\n"
        << " p95  = "
        << r.dpu.p95
        << "\n"
        << " p99  = "
        << r.dpu.p99
        << "\n";
  }

  if (r.preprocess.count) {
    std::cout
        << "\nPREPROCESS mean = "
        << r.preprocess.mean
        << " ms\n";
  }

  if (r.postprocess.count) {
    std::cout
        << "POSTPROCESS mean = "
        << r.postprocess.mean
        << " ms\n";
  }

  if (r.inter_completion.count) {
    std::cout
        << "\nINTER-COMPLETION ms\n"
        << " mean = "
        << r.inter_completion.mean
        << "\n"
        << " p95  = "
        << r.inter_completion.p95
        << "\n"
        << " p99  = "
        << r.inter_completion.p99
        << "\n"
        << "OUTPUT-CADENCE FPS\n"
        << " average(system wall) = "
        << r.throughput_fps
        << "\n"
        << " min = "
        << (r.inter_completion.max > 0.0 ? 1000.0 / r.inter_completion.max : 0.0)
        << "\n"
        << " max = "
        << (r.inter_completion.min > 0.0 ? 1000.0 / r.inter_completion.min : 0.0)
        << "\n"
        << " p95 = "
        << (r.inter_completion.p95 > 0.0 ? 1000.0 / r.inter_completion.p95 : 0.0)
        << "\n"
        << " p99 = "
        << (r.inter_completion.p99 > 0.0 ? 1000.0 / r.inter_completion.p99 : 0.0)
        << "\n";
  }
}

static void write_overhead_csv(
    const Options& o,
    const std::vector<BenchmarkResult>& results) {
  std::ofstream f(fs::path(o.out) / "benchmark_overhead.csv");
  f << "model,dataset,cpu_cores,comparison,model_only_fps,end_to_end_fps,"
       "model_only_mean_ms,end_to_end_mean_ms,overhead_mean_ms,"
       "overhead_mean_pct,throughput_loss_fps,throughput_loss_pct\n";

  const auto write_pair = [&](const std::string& name,
                              const std::string& mo_name,
                              const std::string& e2e_name) {
    const BenchmarkResult* mo = nullptr;
    const BenchmarkResult* e2e = nullptr;
    for (const auto& result : results) {
      if (result.mode == mo_name) mo = &result;
      if (result.mode == e2e_name) e2e = &result;
    }
    if (mo == nullptr || e2e == nullptr) return;
    const double overhead_ms = e2e->latency.mean - mo->latency.mean;
    const double throughput_loss = mo->throughput_fps - e2e->throughput_fps;
    f << std::setprecision(12)
      << fs::path(o.model).filename().string() << ',' << o.dataset << ','
      << o.cpu_cores << ',' << name << ','
      << mo->throughput_fps << ',' << e2e->throughput_fps << ','
      << mo->latency.mean << ',' << e2e->latency.mean << ',' << overhead_ms << ','
      << (mo->latency.mean == 0.0 ? 0.0 : overhead_ms / mo->latency.mean * 100.0) << ','
      << throughput_loss << ','
      << (mo->throughput_fps == 0.0 ? 0.0 : throughput_loss / mo->throughput_fps * 100.0)
      << '\n';
  };

  write_pair("baseline", "baseline_model_only", "baseline_end_to_end");
  write_pair("maximum", "max_model_only_throughput", "max_end_to_end_throughput");
}

static void write_config(
    const Options& o,
    const ModelContext& model) {
  std::ofstream f(
      fs::path(o.out)
      / "config.txt");

  f
      << "platform=ZCU104\n"
      << "precision=INT8\n"
      << "model_only_scope="
      << (model.cpu_subgraphs == 0 && model.dpu_subgraphs == 1
          ? "dpu_execute_async_wait"
          : "graph_execute_async_wait")
      << "\n"
      << "dpu_subgraphs=" << model.dpu_subgraphs << "\n"
      << "cpu_fallback_subgraphs=" << model.cpu_subgraphs << "\n"
      << "batch=1\n"
      << "profile="
      << o.profile
      << "\n"
      << "runners="
      << o.runners
      << "\n"
      << "pre_workers="
      << o.pre_workers
      << "\n"
      << "post_workers="
      << o.post_workers
      << "\n"
      << "slots_per_runner="
      << o.slots_per_runner
      << "\n"
      << "iterations="
      << o.iterations
      << "\n"
      << "warmup="
      << o.warmup
      << "\n"
      << "samples="
      << o.samples
      << "\n"
      << "cpu_cores="
      << o.cpu_cores
      << "\n"
      << "pin="
      << (o.pin ? "true" : "false")
      << "\n"
      << "power=" << (o.power ? "true" : "false") << "\n"
      << "power_sample_ms=" << o.power_sample_ms << "\n"
      << "model="
      << o.model
      << "\n"
      << "dataset="
      << o.dataset
      << "\n"
      << "csv="
      << o.csv
      << "\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
  try {
    cv::setNumThreads(1);
    cv::setUseOptimized(true);

    Options o =
        parse_options(
            argc,
            argv);

    if (o.self_test) {
      run_self_test();
      return 0;
    }

    if (!fs::exists(o.model)) {
      throw std::runtime_error(
          "Model not found: "
          + o.model);
    }

    if (!fs::exists(o.dataset)) {
      throw std::runtime_error(
          "Dataset not found: "
          + o.dataset);
    }

    if (!fs::exists(o.csv)) {
      throw std::runtime_error(
          "CSV not found: "
          + o.csv);
    }

    ensure_dir(o.out);

    limit_process_to_cpus(o.cpu_cores);

    auto dataset =
        load_dataset_csv(o.csv, o.dataset, o.samples);

    if (o.iterations == 0) {
      o.iterations = static_cast<int>(dataset.size());
    }

    std::cout
        << "============================================================\n"
        << "METHANE SEGMENTATION ZCU104 OPTIMIZED BENCHMARK\n"
        << "============================================================\n"
        << "Batch: 1 ALWAYS\n"
        << "Profile: "
        << o.profile
        << "\n"
        << "Dataset images: "
        << dataset.size()
        << "\n"
        << "Runners: "
        << o.runners
        << "\n"
        << "Pre workers: "
        << o.pre_workers
        << "\n"
        << "Post workers: "
        << o.post_workers
        << "\n"
        << "Slots/runner: "
        << o.slots_per_runner
        << "\n"
        << "Pin: "
        << (o.pin ? "yes" : "no")
        << "\n";

    ModelContext model(
        o.model);

    std::cout << "DPU subgraphs: " << model.dpu_subgraphs
              << " | CPU fallback subgraphs: " << model.cpu_subgraphs << "\n";

    write_config(o, model);

    if (o.validate) {
      run_validation(
          o,
          dataset,
          model);
    }

    std::vector<BenchmarkResult> results;

    if (o.profile == "all" ||
        o.profile == "baseline") {
      auto baseline =
          run_baseline(
              o,
              dataset,
              model);

      for (auto& r : baseline) {
        results.push_back(
            std::move(r));
      }
    }

    if (o.profile == "all" ||
        o.profile == "max-model-only") {
      results.push_back(
          run_max_model_only(
              o,
              dataset,
              model));
    }

    if (o.profile == "all" ||
        o.profile == "max-e2e") {
      results.push_back(
          run_max_e2e(
              o,
              dataset,
              model));
    }

    std::ofstream summary(
        fs::path(o.out)
        / "benchmark_summary.csv");

    write_summary_header(
        summary);

    const fs::path samples_path =
        fs::path(o.out)
        / "benchmark_samples.csv";

    if (fs::exists(samples_path)) {
      fs::remove(samples_path);
    }
    const fs::path power_path = fs::path(o.out) / "benchmark_power_rails.csv";
    if (fs::exists(power_path)) fs::remove(power_path);

    for (const auto& r : results) {
      print_result(r);

      write_summary_row(
          summary,
          o,
          r);

      write_samples_csv(
          samples_path,
          r);
      write_power_csv(power_path, r);
    }

    write_overhead_csv(o, results);

    std::cout
        << "\nResults saved to:\n"
        << o.out
        << "\n";

    return 0;
  }
  catch (const std::exception& e) {
    std::cerr
        << "FATAL: "
        << e.what()
        << "\n";

    return 2;
  }
}

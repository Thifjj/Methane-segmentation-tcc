// ============================================================================
// Methane segmentation - staged C++17 sweep controller for ZCU104
//
// Runs benchmark_zcu104 with independent result directories.
// No Python dependency. Batch remains 1 in the benchmark executable.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Options {
  std::string binary = "./benchmark_zcu104";
  std::string model;
  std::string dataset;
  std::string csv;
  std::string out = "benchmark_results";

  int max_concurrency = 16;
  int max_runners = 0;  // 0 = min(2 * DPU cores, max_concurrency)
  int dpu_cores = 2;
  int search_iterations = 24;
  int sweep_samples = 8;
  int full_samples = 0;
  int candidate_iterations = 0;  // 0 = search_iterations
  int final_iterations = 0;      // 0 = one inference per selected image
  int warmup = 20;
  int final_warmup = -1;         // -1 = warmup
  int baseline_repeats = 100;
  int baseline_e2e_passes = 5;
  int final_candidates = 3;
  int final_repeats = 3;
  int power_sample_ms = 200;

  std::string pin_modes = "both";  // both|pin|no-pin
  bool skip_baseline = false;
  bool quick_sweep = false;
  bool resume = false;
  bool dry_run = false;
  bool power = true;
};

struct Config {
  std::string profile;
  int runners = 1;
  int pre_workers = 1;
  int post_workers = 1;
  int slots_per_runner = 1;
  bool pin = false;

  int total_slots() const {
    return runners * slots_per_runner;
  }
};

struct Result {
  std::string stage;
  std::string run_id;
  int repeat = 1;
  Config config;
  int iterations = 0;
  double throughput_fps = 0.0;
  double latency_mean_ms = 0.0;
  double latency_median_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_max_ms = 0.0;
  double latency_p95_ms = 0.0;
  double latency_p99_ms = 0.0;
  double dpu_mean_ms = 0.0;
  double preprocess_mean_ms = 0.0;
  double postprocess_mean_ms = 0.0;
  double io_mean_ms = 0.0;
  double wall_s = 0.0;
  double power_sensor_count = 0.0;
  double power_samples = 0.0;
  double power_mean_w = 0.0;
  double power_min_w = 0.0;
  double power_max_w = 0.0;
  double energy_j = 0.0;
  int samples = 0;
  int cpu_cores = 0;
  std::string status;
  std::string result_dir;
};

static void usage(const char* argv0) {
  std::cout
      << "Methane segmentation ZCU104 staged sweep + full benchmark\n\n"
      << "Usage: " << argv0 << " [options]\n\n"
      << "Paths:\n"
      << "  --binary PATH\n"
      << "  --model PATH\n"
      << "  --dataset DIR\n"
      << "  --csv PATH\n"
      << "  --out DIR\n\n"
      << "Search limits:\n"
      << "  --max-concurrency N   workers and total slots, 1..16\n"
      << "  --max-runners N       0=automatic, otherwise 1..16\n"
      << "  --dpu-cores N         default 2 for this ZCU104\n"
      << "  --pin-modes both|pin|no-pin\n\n"
      << "Benchmark:\n"
      << "  --search-iterations N\n"
      << "  --sweep-samples N      distinct images used by the quick sweep\n"
      << "  --full-samples N       0=all images in the final benchmark\n"
      << "  --candidate-iterations N  0=use search-iterations\n"
      << "  --final-iterations N      0=one inference per selected image\n"
      << "  --warmup N\n"
      << "  --final-warmup N     -1=use warmup\n"
      << "  --baseline-repeats N\n"
      << "  --baseline-e2e-passes N\n"
      << "  --final-candidates N\n"
      << "  --final-repeats N\n\n"
      << "  --power | --no-power\n"
      << "  --power-sample-ms N  default 200 ms\n\n"
      << "Control:\n"
      << "  --skip-baseline\n"
      << "  --quick-sweep        only tune runners and 1-2 pre-workers\n"
      << "  --resume\n"
      << "  --dry-run\n";
}

static Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + a);
      }
      return argv[++i];
    };

    if (a == "--binary") o.binary = value();
    else if (a == "--model") o.model = value();
    else if (a == "--dataset") o.dataset = value();
    else if (a == "--csv") o.csv = value();
    else if (a == "--out") o.out = value();
    else if (a == "--max-concurrency") o.max_concurrency = std::stoi(value());
    else if (a == "--max-runners") o.max_runners = std::stoi(value());
    else if (a == "--dpu-cores") o.dpu_cores = std::stoi(value());
    else if (a == "--search-iterations") o.search_iterations = std::stoi(value());
    else if (a == "--sweep-samples") o.sweep_samples = std::stoi(value());
    else if (a == "--full-samples") o.full_samples = std::stoi(value());
    else if (a == "--candidate-iterations") o.candidate_iterations = std::stoi(value());
    else if (a == "--final-iterations") o.final_iterations = std::stoi(value());
    else if (a == "--warmup") o.warmup = std::stoi(value());
    else if (a == "--final-warmup") o.final_warmup = std::stoi(value());
    else if (a == "--baseline-repeats") o.baseline_repeats = std::stoi(value());
    else if (a == "--baseline-e2e-passes") o.baseline_e2e_passes = std::stoi(value());
    else if (a == "--final-candidates") o.final_candidates = std::stoi(value());
    else if (a == "--final-repeats") o.final_repeats = std::stoi(value());
    else if (a == "--power-sample-ms") o.power_sample_ms = std::stoi(value());
    else if (a == "--power") o.power = true;
    else if (a == "--no-power") o.power = false;
    else if (a == "--pin-modes") o.pin_modes = value();
    else if (a == "--skip-baseline") o.skip_baseline = true;
    else if (a == "--quick-sweep") o.quick_sweep = true;
    else if (a == "--resume") o.resume = true;
    else if (a == "--dry-run") o.dry_run = true;
    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (o.max_concurrency < 1 || o.max_concurrency > 16) {
    throw std::runtime_error("--max-concurrency must be in 1..16");
  }
  if (o.max_runners < 0 || o.max_runners > 16) {
    throw std::runtime_error("--max-runners must be in 0..16");
  }
  if (o.dpu_cores < 1 || o.dpu_cores > 16) {
    throw std::runtime_error("--dpu-cores must be in 1..16");
  }
  if (o.search_iterations <= 0 || o.candidate_iterations < 0 ||
      o.final_iterations < 0 ||
      o.final_candidates <= 0 || o.final_repeats <= 0 ||
      o.power_sample_ms <= 0 ||
      o.baseline_repeats <= 0 || o.baseline_e2e_passes <= 0 ||
      o.warmup < 0 || o.final_warmup < -1 ||
      o.sweep_samples <= 0 || o.full_samples < 0) {
    throw std::runtime_error("Invalid iteration/repetition option");
  }
  if (o.pin_modes != "both" && o.pin_modes != "pin" &&
      o.pin_modes != "no-pin") {
    throw std::runtime_error("--pin-modes must be both, pin or no-pin");
  }
  if (o.model.empty() || o.dataset.empty()) {
    throw std::runtime_error("--model and --dataset are required");
  }
  if (o.csv.empty()) o.csv = (fs::path(o.dataset) / "test.csv").string();
  return o;
}

static std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string current;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(current);
      current.clear();
    } else {
      current.push_back(ch);
    }
  }
  fields.push_back(current);
  return fields;
}

static std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') out += "'\\''";
    else out.push_back(ch);
  }
  out.push_back('\'');
  return out;
}

static std::string json_quote(const std::string& value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    if (ch == '\n') out += "\\n";
    else if (ch != '\r') out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

static std::string config_key(const Config& c) {
  std::ostringstream s;
  s << c.profile << '|' << c.runners << '|' << c.pre_workers << '|'
    << c.post_workers << '|' << c.slots_per_runner << '|' << c.pin;
  return s.str();
}

static std::vector<int> geometric_values(int limit) {
  std::vector<int> values{1};
  while (values.back() < limit) {
    const int next = std::min(limit, values.back() * 2);
    if (next == values.back()) break;
    values.push_back(next);
  }
  return values;
}

static std::vector<bool> pin_values(const std::string& mode) {
  if (mode == "pin") return {true};
  if (mode == "no-pin") return {false};
  return {false, true};
}

static std::vector<Config> unique_configs(const std::vector<Config>& configs) {
  std::set<std::string> seen;
  std::vector<Config> output;
  for (const auto& config : configs) {
    if (seen.insert(config_key(config)).second) output.push_back(config);
  }
  return output;
}

static std::string run_name(
    const std::string& stage,
    const Config& c,
    int repeat,
    int iterations,
    int samples,
    int cpu_cores) {
  std::string profile = c.profile;
  std::replace(profile.begin(), profile.end(), '-', '_');
  std::ostringstream s;
  s << stage << "__" << profile
    << "__r" << c.runners
    << "_pre" << c.pre_workers
    << "_post" << c.post_workers
    << "_spr" << c.slots_per_runner
    << (c.pin ? "_pin" : "_nopin")
    << "__rep" << repeat
    << "__n" << iterations
    << "__samples" << samples
    << "__cpu" << cpu_cores;
  return s.str();
}

static double number(
    const std::map<std::string, std::string>& row,
    const std::string& name) {
  const auto it = row.find(name);
  if (it == row.end() || it->second.empty()) return 0.0;
  try {
    return std::stod(it->second);
  } catch (...) {
    return 0.0;
  }
}

static std::map<std::string, std::string> read_mode_row(
    const fs::path& path,
    const std::string& profile) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open " + path.string());

  std::string expected;
  if (profile == "baseline") expected = "baseline_end_to_end";
  else if (profile == "max-model-only") expected = "max_model_only_throughput";
  else expected = "max_end_to_end_throughput";

  std::string line;
  if (!std::getline(f, line)) throw std::runtime_error("Empty CSV " + path.string());
  const auto header = parse_csv_line(line);
  while (std::getline(f, line)) {
    const auto fields = parse_csv_line(line);
    std::map<std::string, std::string> row;
    for (size_t i = 0; i < header.size() && i < fields.size(); ++i) {
      row[header[i]] = fields[i];
    }
    if (row["mode"] == expected) return row;
  }
  throw std::runtime_error("Mode " + expected + " absent from " + path.string());
}

static double mean_sample_column(
    const fs::path& path,
    const std::string& expected_mode,
    const std::string& column) {
  std::ifstream f(path);
  if (!f) return 0.0;
  std::string line;
  if (!std::getline(f, line)) return 0.0;
  const auto header = parse_csv_line(line);
  int mode_col = -1;
  int value_col = -1;
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == "mode") mode_col = static_cast<int>(i);
    if (header[i] == column) value_col = static_cast<int>(i);
  }
  if (mode_col < 0 || value_col < 0) return 0.0;
  double sum = 0.0;
  size_t count = 0;
  while (std::getline(f, line)) {
    const auto fields = parse_csv_line(line);
    if (static_cast<size_t>(std::max(mode_col, value_col)) >= fields.size()) continue;
    if (fields[mode_col] != expected_mode) continue;
    try {
      sum += std::stod(fields[value_col]);
      ++count;
    } catch (...) {}
  }
  return count ? sum / static_cast<double>(count) : 0.0;
}

static Result execute_one(
    const Options& o,
    const std::string& stage,
    const Config& config,
    int repeat,
    int iterations,
    bool validate,
    int samples = -1,
    int cpu_cores = 0,
    int warmup = -1) {
  if (samples < 0) samples = o.sweep_samples;
  Result result;
  result.stage = stage;
  result.run_id = run_name(stage, config, repeat, iterations, samples, cpu_cores);
  result.repeat = repeat;
  result.config = config;
  result.iterations = iterations;
  result.samples = samples;
  result.cpu_cores = cpu_cores;
  result.result_dir = (fs::path(o.out) / "runs" / result.run_id).string();

  const fs::path result_dir(result.result_dir);
  const fs::path summary = result_dir / "benchmark_summary.csv";
  const fs::path samples_csv = result_dir / "benchmark_samples.csv";
  const fs::path log = result_dir / "execution.log";

  std::ostringstream command;
  command
      << shell_quote(o.binary)
      << " --profile " << shell_quote(config.profile)
      << " --model " << shell_quote(o.model)
      << " --dataset " << shell_quote(o.dataset)
      << " --csv " << shell_quote(o.csv)
      << " --out " << shell_quote(result.result_dir)
      << " --runners " << config.runners
      << " --pre-workers " << config.pre_workers
      << " --post-workers " << config.post_workers
      << " --slots-per-runner " << config.slots_per_runner
      << " --iterations " << iterations
      << " --samples " << samples
      << " --cpu-cores " << cpu_cores
      << " --warmup " << (warmup >= 0 ? warmup : o.warmup)
      << " --baseline-repeats " << o.baseline_repeats
      << " --baseline-e2e-passes " << o.baseline_e2e_passes
      << (config.pin ? " --pin" : " --no-pin")
      << (validate ? " --validate" : " --no-validate");
  command << (o.power ? " --power" : " --no-power")
          << " --power-sample-ms " << o.power_sample_ms;

  std::cout << "\n[" << stage << "] " << result.run_id << "\n"
            << "  " << command.str() << "\n";

  if (o.dry_run) {
    result.status = "dry-run";
    return result;
  }

  if (o.resume && fs::exists(summary)) {
    result.status = "reused";
  } else {
    fs::create_directories(result_dir);
    std::ofstream metadata(result_dir / "sweep_command.txt");
    metadata << command.str() << "\n";
    metadata.close();

    const std::string redirected =
        command.str() + " > " + shell_quote(log.string()) + " 2>&1";
    const auto started = std::chrono::steady_clock::now();
    const int rc = std::system(redirected.c_str());
    const auto ended = std::chrono::steady_clock::now();
    result.wall_s = std::chrono::duration<double>(ended - started).count();
    if (rc != 0) {
      result.status = "failed";
      std::cerr << "  ERROR: command returned " << rc
                << "; see " << log << "\n";
      return result;
    }
    result.status = "ok";
  }

  const auto row = read_mode_row(summary, config.profile);
  const std::string mode = row.at("mode");
  result.throughput_fps = number(row, "throughput_fps");
  result.latency_mean_ms = number(row, "latency_mean_ms");
  result.latency_median_ms = number(row, "latency_median_ms");
  result.latency_min_ms = number(row, "latency_min_ms");
  result.latency_max_ms = number(row, "latency_max_ms");
  result.latency_p95_ms = number(row, "latency_p95_ms");
  result.latency_p99_ms = number(row, "latency_p99_ms");
  result.dpu_mean_ms = number(row, "dpu_mean_ms");
  result.preprocess_mean_ms = number(row, "preprocess_mean_ms");
  result.postprocess_mean_ms = number(row, "postprocess_mean_ms");
  result.wall_s = number(row, "wall_s");
  result.power_sensor_count = number(row, "power_sensor_count");
  result.power_samples = number(row, "power_samples");
  result.power_mean_w = number(row, "power_mean_w");
  result.power_min_w = number(row, "power_min_w");
  result.power_max_w = number(row, "power_max_w");
  result.energy_j = number(row, "energy_j");
  result.iterations = static_cast<int>(number(row, "completed"));
  result.io_mean_ms = mean_sample_column(samples_csv, mode, "io_ms");

  std::cout << std::fixed << std::setprecision(4)
            << "  FPS=" << result.throughput_fps
            << " latency=" << result.latency_mean_ms
            << " ms P99=" << result.latency_p99_ms << " ms\n";
  return result;
}

static bool valid(const Result& r) {
  return r.status == "ok" || r.status == "reused";
}

static bool performance_order(const Result& a, const Result& b) {
  if (a.throughput_fps != b.throughput_fps) {
    return a.throughput_fps > b.throughput_fps;
  }
  return a.latency_p99_ms < b.latency_p99_ms;
}

static Result best_result(const std::vector<Result>& results) {
  std::vector<Result> candidates;
  for (const auto& result : results) if (valid(result)) candidates.push_back(result);
  if (candidates.empty()) throw std::runtime_error("No valid result in stage");
  std::sort(candidates.begin(), candidates.end(), performance_order);
  return candidates.front();
}

static std::vector<int> top_unique_values(
    std::vector<Result> results,
    const std::string& attribute,
    int count = 2) {
  results.erase(
      std::remove_if(results.begin(), results.end(),
                     [](const Result& r) { return !valid(r); }),
      results.end());
  std::sort(results.begin(), results.end(), performance_order);
  std::vector<int> values;
  for (const auto& r : results) {
    int value = 0;
    if (attribute == "runners") value = r.config.runners;
    else if (attribute == "pre_workers") value = r.config.pre_workers;
    else if (attribute == "post_workers") value = r.config.post_workers;
    else if (attribute == "slots_per_runner") value = r.config.slots_per_runner;
    if (std::find(values.begin(), values.end(), value) == values.end()) {
      values.push_back(value);
    }
    if (static_cast<int>(values.size()) == count) break;
  }
  return values;
}

static double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2) return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

static void write_results_csv(
    const fs::path& path,
    const Options& o,
    const std::vector<Result>& results) {
  std::ofstream f(path);
  f << "model,dataset,csv,sweep_samples,search_iterations,candidate_iterations,max_concurrency,"
    << "max_runners,dpu_cores,warmup,final_warmup,final_repeats,quick_sweep,"
    << "stage,run_id,repeat,profile,"
    << "runners,pre_workers,post_workers,"
    << "slots_per_runner,total_slots,pin,iterations,samples,cpu_cores,throughput_fps,"
    << "latency_mean_ms,latency_median_ms,latency_min_ms,latency_max_ms,"
    << "latency_p95_ms,latency_p99_ms,dpu_mean_ms,"
    << "preprocess_mean_ms,postprocess_mean_ms,io_mean_ms,wall_s,"
    << "power_sensor_count,power_samples,power_mean_w,power_min_w,power_max_w,energy_j,"
    << "status,result_dir\n";
  f << std::setprecision(12);
  for (const auto& r : results) {
    f << '"' << fs::path(o.model).filename().string() << "\",\""
      << o.dataset << "\",\"" << o.csv << "\","
      << o.sweep_samples << ',' << o.search_iterations << ','
      << (o.candidate_iterations > 0 ? o.candidate_iterations : o.search_iterations) << ','
      << o.max_concurrency << ',' << o.max_runners << ',' << o.dpu_cores << ','
      << o.warmup << ','
      << (o.final_warmup >= 0 ? o.final_warmup : o.warmup) << ','
      << o.final_repeats << ',' << (o.quick_sweep ? "true" : "false") << ','
      << r.stage << ',' << r.run_id << ',' << r.repeat << ','
      << r.config.profile << ',' << r.config.runners << ','
      << r.config.pre_workers << ',' << r.config.post_workers << ','
      << r.config.slots_per_runner << ',' << r.config.total_slots() << ','
      << (r.config.pin ? "true" : "false") << ',' << r.iterations << ','
      << r.samples << ',' << r.cpu_cores << ','
      << r.throughput_fps << ',' << r.latency_mean_ms << ','
      << r.latency_median_ms << ',' << r.latency_min_ms << ','
      << r.latency_max_ms << ','
      << r.latency_p95_ms << ',' << r.latency_p99_ms << ','
      << r.dpu_mean_ms << ',' << r.preprocess_mean_ms << ','
      << r.postprocess_mean_ms << ',' << r.io_mean_ms << ',' << r.wall_s << ','
      << r.power_sensor_count << ',' << r.power_samples << ',' << r.power_mean_w << ','
      << r.power_min_w << ',' << r.power_max_w << ',' << r.energy_j << ','
      << r.status << ',' << '"' << r.result_dir << '"' << '\n';
  }
}

static void write_campaign(
    const Options& o,
    int max_runners,
    const std::vector<int>& runner_values,
    const std::vector<int>& worker_values) {
  std::ofstream f(fs::path(o.out) / "campaign.txt");
  f << "language=C++17\n"
    << "dpu_cores=" << o.dpu_cores << "\n"
    << "max_runners=" << max_runners << "\n"
    << "max_concurrency=" << o.max_concurrency << "\n"
    << "search_iterations=" << o.search_iterations << "\n"
    << "candidate_iterations="
    << (o.candidate_iterations > 0 ? o.candidate_iterations : o.search_iterations) << "\n"
    << "sweep_samples=" << o.sweep_samples << "\n"
    << "full_samples=" << o.full_samples << "\n"
    << "final_iterations=" << o.final_iterations << "\n"
    << "final_candidates=" << o.final_candidates << "\n"
    << "final_repeats=" << o.final_repeats << "\n"
    << "quick_sweep=" << (o.quick_sweep ? "true" : "false") << "\n"
    << "final_warmup=" << (o.final_warmup >= 0 ? o.final_warmup : o.warmup) << "\n"
    << "power=" << (o.power ? "true" : "false") << "\n"
    << "power_sample_ms=" << o.power_sample_ms << "\n"
    << "pin_modes=" << o.pin_modes << "\nrunner_values=";
  for (size_t i = 0; i < runner_values.size(); ++i) {
    if (i) f << ':';
    f << runner_values[i];
  }
  f << "\nworker_values=";
  for (size_t i = 0; i < worker_values.size(); ++i) {
    if (i) f << ':';
    f << worker_values[i];
  }
  f << '\n';
}

static Result write_best(
    const Options& o,
    const std::vector<Result>& final_results,
    const Result& fallback,
    const Result& best_model_only) {
  std::map<std::string, std::vector<Result>> groups;
  for (const auto& result : final_results) {
    if (valid(result)) groups[config_key(result.config)].push_back(result);
  }

  Result chosen = fallback;
  double chosen_fps = fallback.throughput_fps;
  double chosen_p99 = fallback.latency_p99_ms;
  bool have_final = false;
  for (const auto& entry : groups) {
    std::vector<double> fps;
    std::vector<double> p99;
    for (const auto& result : entry.second) {
      fps.push_back(result.throughput_fps);
      p99.push_back(result.latency_p99_ms);
    }
    const double mfps = median(fps);
    const double mp99 = median(p99);
    if (!have_final || mfps > chosen_fps ||
        (mfps == chosen_fps && mp99 < chosen_p99)) {
      have_final = true;
      chosen = entry.second.front();
      chosen_fps = mfps;
      chosen_p99 = mp99;
    }
  }

  std::ofstream json(fs::path(o.out) / "best_config.json");
  json << std::setprecision(12)
       << "{\n"
       << "  \"selection\": \"highest median final throughput; P99 breaks ties\",\n"
       << "  \"model\": " << json_quote(fs::path(o.model).filename().string()) << ",\n"
       << "  \"dataset\": " << json_quote(o.dataset) << ",\n"
       << "  \"sweep_samples\": " << o.sweep_samples << ",\n"
       << "  \"search_iterations\": " << o.search_iterations << ",\n"
       << "  \"candidate_iterations\": "
       << (o.candidate_iterations > 0 ? o.candidate_iterations : o.search_iterations)
       << ",\n"
       << "  \"quick_sweep\": " << (o.quick_sweep ? "true" : "false") << ",\n"
       << "  \"median_throughput_fps\": " << chosen_fps << ",\n"
       << "  \"median_latency_p99_ms\": " << chosen_p99 << ",\n"
       << "  \"model_only_config\": {\n"
       << "    \"runners\": " << best_model_only.config.runners << ",\n"
       << "    \"pin\": " << (best_model_only.config.pin ? "true" : "false") << "\n"
       << "  },\n"
       << "  \"end_to_end_config\": {\n"
       << "    \"profile\": \"max-e2e\",\n"
       << "    \"runners\": " << chosen.config.runners << ",\n"
       << "    \"pre_workers\": " << chosen.config.pre_workers << ",\n"
       << "    \"post_workers\": " << chosen.config.post_workers << ",\n"
       << "    \"slots_per_runner\": " << chosen.config.slots_per_runner << ",\n"
       << "    \"total_slots\": " << chosen.config.total_slots() << ",\n"
       << "    \"pin\": " << (chosen.config.pin ? "true" : "false") << "\n"
       << "  }\n}\n";

  const fs::path reproduce = fs::path(o.out) / "reproduce_best.sh";
  std::ofstream script(reproduce);
  script << "#!/bin/bash\nset -e\n\n"
         << shell_quote(o.binary)
         << " --profile max-e2e"
         << " --model " << shell_quote(o.model)
         << " --dataset " << shell_quote(o.dataset)
         << " --csv " << shell_quote(o.csv)
         << " --out " << shell_quote((fs::path(o.out) / "best_reproduction").string())
         << " --runners " << chosen.config.runners
         << " --pre-workers " << chosen.config.pre_workers
         << " --post-workers " << chosen.config.post_workers
         << " --slots-per-runner " << chosen.config.slots_per_runner
         << " --iterations " << o.final_iterations
         << " --samples " << o.full_samples
         << " --cpu-cores 4"
         << " --warmup " << o.warmup
         << (chosen.config.pin ? " --pin" : " --no-pin")
         << " --no-validate\n";
  script.close();
  fs::permissions(
      reproduce,
      fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
      fs::perm_options::add);
  return chosen;
}

static void write_overhead_csv(
    const Options& o,
    const std::vector<Result>& model_only,
    const std::vector<Result>& end_to_end) {
  std::ofstream f(fs::path(o.out) / "benchmark_overhead.csv");
  f << "model,dataset,sweep_samples,search_iterations,full_samples,full_iterations,"
       "repeat,cpu_cores,model_only_runners,end_to_end_runners,pre_workers,"
       "post_workers,slots_per_runner,pin,model_only_fps,end_to_end_fps,"
       "model_only_mean_ms,end_to_end_mean_ms,overhead_mean_ms,overhead_mean_pct,"
       "overhead_median_ms,overhead_p95_ms,overhead_p99_ms,total_time_overhead_s,"
       "throughput_loss_fps,throughput_loss_pct\n";
  f << std::setprecision(12);
  for (const auto& mo : model_only) {
    const auto found = std::find_if(end_to_end.begin(), end_to_end.end(),
        [&](const Result& e2e) {
          return e2e.repeat == mo.repeat && e2e.cpu_cores == mo.cpu_cores;
        });
    if (found == end_to_end.end() || !valid(mo) || !valid(*found)) continue;
    const auto& e2e = *found;
    const double overhead = e2e.latency_mean_ms - mo.latency_mean_ms;
    const double loss = mo.throughput_fps - e2e.throughput_fps;
    f << '"' << fs::path(o.model).filename().string() << "\",\""
      << o.dataset << "\"," << o.sweep_samples << ',' << o.search_iterations << ','
      << o.full_samples << ',' << mo.iterations << ',' << mo.repeat << ','
      << mo.cpu_cores << ',' << mo.config.runners << ',' << e2e.config.runners << ','
      << e2e.config.pre_workers << ',' << e2e.config.post_workers << ','
      << e2e.config.slots_per_runner << ',' << (e2e.config.pin ? "true" : "false") << ','
      << mo.throughput_fps << ',' << e2e.throughput_fps << ','
      << mo.latency_mean_ms << ',' << e2e.latency_mean_ms << ',' << overhead << ','
      << (mo.latency_mean_ms == 0.0 ? 0.0 : overhead / mo.latency_mean_ms * 100.0) << ','
      << (e2e.latency_median_ms - mo.latency_median_ms) << ','
      << (e2e.latency_p95_ms - mo.latency_p95_ms) << ','
      << (e2e.latency_p99_ms - mo.latency_p99_ms) << ','
      << (e2e.wall_s - mo.wall_s) << ','
      << loss << ','
      << (mo.throughput_fps == 0.0 ? 0.0 : loss / mo.throughput_fps * 100.0)
      << '\n';
  }
}

int main(int argc, char** argv) {
  try {
    const Options o = parse_options(argc, argv);
    if (!o.dry_run) {
      if (!fs::exists(o.binary)) throw std::runtime_error("Binary not found: " + o.binary);
      if (!fs::exists(o.model)) throw std::runtime_error("Model not found: " + o.model);
      if (!fs::exists(o.dataset)) throw std::runtime_error("Dataset not found: " + o.dataset);
      if (!fs::exists(o.csv)) throw std::runtime_error("CSV not found: " + o.csv);
    }
    fs::create_directories(o.out);

    const int max_runners = o.max_runners > 0
        ? std::min({o.max_runners, o.max_concurrency, 16})
        : std::min({2 * o.dpu_cores, o.max_concurrency, 16});
    std::vector<int> runner_values;
    for (int runner = 1; runner <= max_runners; ++runner) {
      runner_values.push_back(runner);
    }
    const auto worker_values = o.quick_sweep
        ? std::vector<int>{1, std::min(2, o.max_concurrency)}
        : geometric_values(o.max_concurrency);
    const auto pins = pin_values(o.pin_modes);
    write_campaign(o, max_runners, runner_values, worker_values);

    std::cout << "C++17 sweep: DPU cores=" << o.dpu_cores
              << " max runners=" << max_runners
              << " max workers/slots=" << o.max_concurrency << "\n";

    std::vector<Result> all_results;
    if (!o.skip_baseline) {
      Config baseline{"baseline", 1, 1, 1, 1, false};
      all_results.push_back(execute_one(
          o, "00_baseline", baseline, 1, o.search_iterations, true));
    }

    std::vector<Result> model_results;
    for (int runners : runner_values) {
      for (bool pin : pins) {
        Config config{"max-model-only", runners, 1, 1, 1, pin};
        model_results.push_back(execute_one(
            o, "10_model_runners", config, 1, o.search_iterations, false));
      }
    }
    all_results.insert(all_results.end(), model_results.begin(), model_results.end());
    if (o.dry_run) {
      write_results_csv(fs::path(o.out) / "all_runs.csv", o, all_results);
      std::cout << "\nDry-run complete; adaptive stages need real measurements.\n";
      return 0;
    }

    const auto top_runners = top_unique_values(model_results, "runners", 2);
    if (top_runners.empty()) throw std::runtime_error("Model-only sweep failed");

    std::vector<Result> seed_results;
    for (int runners : top_runners) {
      for (bool pin : pins) {
        const int spr = o.quick_sweep
            ? 1
            : std::max(1, std::min(2, o.max_concurrency / runners));
        Config config{"max-e2e", runners, std::min(2, o.max_concurrency), 1, spr, pin};
        seed_results.push_back(execute_one(
            o, "20_e2e_runners", config, 1, o.search_iterations, false));
      }
    }
    all_results.insert(all_results.end(), seed_results.begin(), seed_results.end());
    Result current = best_result(seed_results);

    std::vector<Result> pre_results;
    for (int pre : worker_values) {
      Config config = current.config;
      config.pre_workers = pre;
      pre_results.push_back(execute_one(
          o, "30_pre_workers", config, 1, o.search_iterations, false));
    }
    all_results.insert(all_results.end(), pre_results.begin(), pre_results.end());
    pre_results.push_back(current);
    current = best_result(pre_results);

    if (!o.quick_sweep) {
    std::vector<Result> post_results;
    for (int post : worker_values) {
      Config config = current.config;
      config.post_workers = post;
      post_results.push_back(execute_one(
          o, "40_post_workers", config, 1, o.search_iterations, false));
    }
    all_results.insert(all_results.end(), post_results.begin(), post_results.end());
    post_results.push_back(current);
    current = best_result(post_results);

    std::vector<Result> slot_results;
    for (int total : geometric_values(o.max_concurrency)) {
      if (total < current.config.runners) continue;
      const int spr = std::max(1, total / current.config.runners);
      Config config = current.config;
      config.slots_per_runner = spr;
      if (config.total_slots() <= o.max_concurrency) {
        slot_results.push_back(execute_one(
            o, "50_slots", config, 1, o.search_iterations, false));
      }
    }
    all_results.insert(all_results.end(), slot_results.begin(), slot_results.end());
    slot_results.push_back(current);
    current = best_result(slot_results);

    auto refine_runners = top_unique_values(seed_results, "runners", 2);
    auto refine_pre = top_unique_values(pre_results, "pre_workers", 2);
    auto refine_post = top_unique_values(post_results, "post_workers", 2);
    auto refine_slots = top_unique_values(slot_results, "slots_per_runner", 2);
    if (refine_runners.empty()) refine_runners = {current.config.runners};
    if (refine_pre.empty()) refine_pre = {current.config.pre_workers};
    if (refine_post.empty()) refine_post = {current.config.post_workers};
    if (refine_slots.empty()) refine_slots = {current.config.slots_per_runner};

    std::vector<Config> refine_configs;
    for (int runners : refine_runners) {
      for (int pre : refine_pre) {
        for (int post : refine_post) {
          for (int spr : refine_slots) {
            Config config{"max-e2e", runners, pre, post, spr, current.config.pin};
            if (config.total_slots() <= o.max_concurrency) refine_configs.push_back(config);
          }
        }
      }
    }
    refine_configs = unique_configs(refine_configs);
    std::vector<Result> refine_results;
    for (const auto& config : refine_configs) {
      refine_results.push_back(execute_one(
          o, "55_refine", config, 1, o.search_iterations, false));
    }
    all_results.insert(all_results.end(), refine_results.begin(), refine_results.end());
    refine_results.push_back(current);
    current = best_result(refine_results);
    }

    std::vector<Result> search_e2e;
    for (const auto& result : all_results) {
      if (result.config.profile == "max-e2e" && valid(result)) search_e2e.push_back(result);
    }
    std::sort(search_e2e.begin(), search_e2e.end(), performance_order);

    std::vector<Config> candidates;
    for (const auto& result : search_e2e) {
      for (bool pin : pins) {
        Config config = result.config;
        config.pin = pin;
        candidates.push_back(config);
      }
      candidates = unique_configs(candidates);
      if (static_cast<int>(candidates.size()) >= o.final_candidates) break;
    }
    if (static_cast<int>(candidates.size()) > o.final_candidates) {
      candidates.resize(static_cast<size_t>(o.final_candidates));
    }

    std::vector<Result> final_results;
    for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
      for (int repeat = 1; repeat <= o.final_repeats; ++repeat) {
        final_results.push_back(execute_one(
            o,
            "60_final_c" + std::to_string(candidate + 1),
            candidates[candidate],
            repeat,
            o.candidate_iterations > 0
                ? o.candidate_iterations
                : o.search_iterations,
            false));
      }
    }
    all_results.insert(all_results.end(), final_results.begin(), final_results.end());

    write_results_csv(fs::path(o.out) / "all_runs.csv", o, all_results);
    write_results_csv(fs::path(o.out) / "ranking_search.csv", o, search_e2e);
    std::sort(final_results.begin(), final_results.end(), performance_order);
    write_results_csv(fs::path(o.out) / "ranking_final_runs.csv", o, final_results);
    const Result best_model_only = best_result(model_results);
    const Result best_e2e = write_best(o, final_results, current, best_model_only);

    std::vector<Result> full_model_only;
    std::vector<Result> full_end_to_end;
    for (int cpu_cores = 1; cpu_cores <= 4; ++cpu_cores) {
      for (int repeat = 1; repeat <= o.final_repeats; ++repeat) {
        full_model_only.push_back(execute_one(
            o, "70_full_model_only", best_model_only.config, repeat,
            o.final_iterations, false, o.full_samples, cpu_cores,
            o.final_warmup >= 0 ? o.final_warmup : o.warmup));
        const bool validate = cpu_cores == 1 && repeat == 1;
        full_end_to_end.push_back(execute_one(
            o, "80_full_end_to_end", best_e2e.config, repeat,
            o.final_iterations, validate, o.full_samples, cpu_cores,
            o.final_warmup >= 0 ? o.final_warmup : o.warmup));
      }
    }

    all_results.insert(all_results.end(), full_model_only.begin(), full_model_only.end());
    all_results.insert(all_results.end(), full_end_to_end.begin(), full_end_to_end.end());
    write_results_csv(fs::path(o.out) / "all_runs.csv", o, all_results);
    std::vector<Result> full_results = full_model_only;
    full_results.insert(full_results.end(), full_end_to_end.begin(), full_end_to_end.end());
    write_results_csv(fs::path(o.out) / "benchmark_geral.csv", o, full_results);
    write_overhead_csv(o, full_model_only, full_end_to_end);

    for (const auto& result : full_end_to_end) {
      const fs::path metrics = fs::path(result.result_dir) / "metricas_globais.csv";
      const fs::path per_image = fs::path(result.result_dir) / "metricas_por_imagem.csv";
      if (fs::exists(metrics)) {
        fs::copy_file(metrics, fs::path(o.out) / "metricas_globais.csv",
                      fs::copy_options::overwrite_existing);
      }
      if (fs::exists(per_image)) {
        fs::copy_file(per_image, fs::path(o.out) / "metricas_por_imagem.csv",
                      fs::copy_options::overwrite_existing);
      }
    }

    std::cout << "\nSweep complete.\n"
              << "Results: " << o.out << "\n"
              << "Best: " << (fs::path(o.out) / "best_config.json") << "\n"
              << "Full 1/2/3/4-core benchmark: "
              << (fs::path(o.out) / "benchmark_geral.csv") << "\n"
              << "Overhead: " << (fs::path(o.out) / "benchmark_overhead.csv") << "\n"
              << "Reproduce: " << (fs::path(o.out) / "reproduce_best.sh") << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 2;
  }
}

 #include "xmodel_runner.hpp"

  #include <cmath>
  #include <cstdlib>
  #include <cstring>
  #include <new>
  #include <stdexcept>
  #include <utility>

  #include <vart/tensor_buffer.hpp>
  #include <vitis/ai/graph_runner.hpp>
  #include <xir/graph/subgraph.hpp>

  namespace {

  class MemoriaAlinhada {
  public:
      explicit MemoriaAlinhada(std::size_t bytes) : bytes_(bytes) {
          void* p = nullptr;
          if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr) {
              throw std::bad_alloc();
          }
          dados_.reset(static_cast<std::uint8_t*>(p));
          std::memset(dados_.get(), 0, bytes);
      }

      std::uint8_t* dados() { return dados_.get(); }
      const std::uint8_t* dados() const { return dados_.get(); }
      std::size_t bytes() const { return bytes_; }

  private:
      struct Liberar {
          void operator()(std::uint8_t* p) const { std::free(p); }
      };

      std::unique_ptr<std::uint8_t, Liberar> dados_;
      std::size_t bytes_;
  };

  class BufferCpu final : public vart::TensorBuffer {
  public:
      BufferCpu(void* memoria, const xir::Tensor* tensor)
          : vart::TensorBuffer(tensor),
            memoria_(static_cast<std::uint8_t*>(memoria)) {}

      std::pair<std::uint64_t, std::size_t> data(
          const std::vector<int> indice = {}
      ) override {
          const auto shape = tensor_->get_shape();
          std::size_t deslocamento = 0;

          if (!indice.empty()) {
              if (indice.size() != shape.size()) {
                  throw std::runtime_error("Índice do tensor com dimensão incorreta");
              }
              for (std::size_t i = 0; i < indice.size(); ++i) {
                  if (indice[i] < 0 || indice[i] >= shape[i]) {
                      throw std::runtime_error("Índice fora do tensor");
                  }
                  deslocamento =
                      deslocamento * static_cast<std::size_t>(shape[i]) +
                      static_cast<std::size_t>(indice[i]);
              }
          }

          const std::size_t total = tensor_->get_data_size();
          const std::size_t elementos = tensor_->get_element_num();
          const std::size_t bytes_por_elemento = total / elementos;
          const std::size_t offset_bytes = deslocamento * bytes_por_elemento;

          return {
              reinterpret_cast<std::uint64_t>(memoria_ + offset_bytes),
              total - offset_bytes
          };
      }

  private:
      std::uint8_t* memoria_;
  };

  struct BufferProprio {
      explicit BufferProprio(const xir::Tensor* tensor)
          : memoria(tensor->get_data_size()),
            buffer(memoria.dados(), tensor) {}

      MemoriaAlinhada memoria;
      BufferCpu buffer;
  };

  void contar_subgrafos(
      const xir::Subgraph* subgrafo,
      int& dpu,
      int& cpu,
      const xir::Subgraph*& primeiro_dpu
  ) {
      if (subgrafo->has_attr("device")) {
          const auto dispositivo = subgrafo->get_attr<std::string>("device");
          if (dispositivo == "DPU") {
              ++dpu;
              if (primeiro_dpu == nullptr) primeiro_dpu = subgrafo;
          } else if (dispositivo == "CPU") {
              ++cpu;
          }
      }

      for (const auto* filho : subgrafo->children_topological_sort()) {
          contar_subgrafos(filho, dpu, cpu, primeiro_dpu);
      }
  }

  int fix_point(const xir::Tensor* tensor, int padrao) {
      for (const char* nome : {"fix_point", "fixpos"}) {
          try {
              if (tensor->has_attr(nome)) return tensor->get_attr<int>(nome);
          } catch (...) {
              // Alguns tensores não expõem esse atributo como inteiro.
          }
      }
      return padrao;
  }

  } // namespace

  struct SlotXModel::Impl {
      Impl(const xir::Tensor* tensor_entrada, const xir::Tensor* tensor_saida)
          : entrada(tensor_entrada),
            saida(tensor_saida),
            entradas{&entrada.buffer},
            saidas{&saida.buffer} {}

      BufferProprio entrada;
      BufferProprio saida;
      std::vector<vart::TensorBuffer*> entradas;
      std::vector<vart::TensorBuffer*> saidas;
  };

  SlotXModel::SlotXModel(const xir::Tensor* entrada, const xir::Tensor* saida)
      : impl_(std::make_unique<Impl>(entrada, saida)) {}

  SlotXModel::~SlotXModel() = default;

  std::int8_t* SlotXModel::dados_entrada() {
      return reinterpret_cast<std::int8_t*>(impl_->entrada.memoria.dados());
  }

  const void* SlotXModel::dados_saida() const {
      return impl_->saida.memoria.dados();
  }

  std::size_t SlotXModel::bytes_entrada() const {
      return impl_->entrada.memoria.bytes();
  }

  std::size_t SlotXModel::bytes_saida() const {
      return impl_->saida.memoria.bytes();
  }

  void XModelRunner::sincronizar_entrada(SlotXModel& slot) {
      slot.impl_->entrada.buffer.sync_for_write(0, slot.bytes_entrada());
  }

  void XModelRunner::inferir(SlotXModel& slot) {
      auto job = runner->execute_async(
          slot.impl_->entradas,
          slot.impl_->saidas
      );
      if (job.second != 0) {
          throw std::runtime_error("execute_async falhou");
      }

      int status = runner->wait(static_cast<int>(job.first), -1);
      if (status != 0) {
          throw std::runtime_error(
              "VART wait falhou: " + std::to_string(status)
          );
      }
  }

  void XModelRunner::sincronizar_saida(SlotXModel& slot) {
      slot.impl_->saida.buffer.sync_for_read(0, slot.bytes_saida());
  }

  std::vector<XModelRunner> carregar_runners(
      const std::string& caminho_xmodel,
      int quantidade,
      int slots_por_runner
  ) {
      if (quantidade < 1 || quantidade > 4 || slots_por_runner < 1) {
          throw std::runtime_error("Quantidade de runners ou slots inválida");
      }

      auto graph_unico = xir::Graph::deserialize(caminho_xmodel);
      if (!graph_unico) {
          throw std::runtime_error("Não foi possível carregar: " + caminho_xmodel);
      }
      std::shared_ptr<xir::Graph> graph(std::move(graph_unico));

      int dpu = 0;
      int cpu = 0;
      const xir::Subgraph* primeiro_dpu = nullptr;

      for (const auto* filho :
           graph->get_root_subgraph()->children_topological_sort()) {
          contar_subgrafos(filho, dpu, cpu, primeiro_dpu);
      }
      if (dpu == 0) {
          throw std::runtime_error("XModel sem subgrafo DPU");
      }

      std::vector<XModelRunner> resultado;
      resultado.reserve(quantidade);

      for (int i = 0; i < quantidade; ++i) {
          XModelRunner contexto;
          contexto.graph = graph;
          contexto.subgrafos_dpu = dpu;
          contexto.subgrafos_cpu = cpu;

          if (dpu == 1 && cpu == 0) {
              contexto.runner = vart::Runner::create_runner(primeiro_dpu, "run");
          } else {
              contexto.attrs = xir::Attrs::create();
              contexto.runner = vitis::ai::GraphRunner::create_graph_runner(
                  graph.get(), contexto.attrs.get()
              );
          }
          if (!contexto.runner) {
              throw std::runtime_error("Falha ao criar runner VART");
          }

          auto entradas = contexto.runner->get_input_tensors();
          auto saidas = contexto.runner->get_output_tensors();
          if (entradas.size() != 1 || saidas.size() != 1) {
              throw std::runtime_error("Esperado um tensor de entrada e um de saída");
          }

          const auto* tensor_entrada = entradas[0];
          const auto* tensor_saida = saidas[0];

          if (tensor_entrada->get_shape() !=
                  std::vector<std::int32_t>{1, 512, 512, 4} ||
              tensor_entrada->get_data_size() != 512 * 512 * 4) {
              throw std::runtime_error("Entrada deve ser INT8 [1,512,512,4]");
          }

          if (tensor_saida->get_shape() !=
              std::vector<std::int32_t>{1, 512, 512, 1}) {
              throw std::runtime_error("Saída deve ser [1,512,512,1]");
          }

          const std::size_t bytes_saida = tensor_saida->get_data_size();
          if (bytes_saida != 512 * 512 &&
              bytes_saida != 512 * 512 * sizeof(float)) {
              throw std::runtime_error("Saída deve ser INT8 ou FLOAT32");
          }

          contexto.saida_float = bytes_saida == 512 * 512 * sizeof(float);
          contexto.escala_entrada =
              std::exp2(static_cast<float>(fix_point(tensor_entrada, 5)));
          contexto.escala_saida =
              std::exp2(-static_cast<float>(fix_point(tensor_saida, 2)));

          for (int s = 0; s < slots_por_runner; ++s) {
              contexto.slots.push_back(
                  std::make_unique<SlotXModel>(tensor_entrada, tensor_saida)
              );
          }

          resultado.push_back(std::move(contexto));
      }

      return resultado;
  }

  XModelRunner carregar_xmodel(const std::string& caminho_xmodel) {
      auto runners = carregar_runners(caminho_xmodel, 1, 1);
      return std::move(runners.front());
  }
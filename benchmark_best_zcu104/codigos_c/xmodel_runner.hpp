  #pragma once

  #include <cstddef>
  #include <cstdint>
  #include <memory>
  #include <string>
  #include <vector>

  #include <vart/runner.hpp>
  #include <xir/attrs/attrs.hpp>
  #include <xir/graph/graph.hpp>
  #include <xir/tensor/tensor.hpp>

  class SlotXModel {
  public:
      SlotXModel(const xir::Tensor* entrada, const xir::Tensor* saida);
      ~SlotXModel();

      SlotXModel(const SlotXModel&) = delete;
      SlotXModel& operator=(const SlotXModel&) = delete;

      std::int8_t* dados_entrada();
      const void* dados_saida() const;
      std::size_t bytes_entrada() const;
      std::size_t bytes_saida() const;

  private:
      struct Impl;
      std::unique_ptr<Impl> impl_;

      friend struct XModelRunner;
  };

  struct XModelRunner {
      std::shared_ptr<xir::Graph> graph;
      std::unique_ptr<xir::Attrs> attrs;
      std::unique_ptr<vart::Runner> runner;
      std::vector<std::unique_ptr<SlotXModel>> slots;

      float escala_entrada = 0.0f;
      float escala_saida = 0.0f;
      bool saida_float = false;
      int subgrafos_dpu = 0;
      int subgrafos_cpu = 0;

      void sincronizar_entrada(SlotXModel& slot);
      void inferir(SlotXModel& slot);
      void sincronizar_saida(SlotXModel& slot);
  };

  // Mantém o uso simples do benchmark_vitis.cpp existente.
  XModelRunner carregar_xmodel(const std::string& caminho_xmodel);

  // Cria 1–4 runners independentes que compartilham o grafo carregado.
  std::vector<XModelRunner> carregar_runners(
      const std::string& caminho_xmodel,
      int quantidade,
      int slots_por_runner
  );
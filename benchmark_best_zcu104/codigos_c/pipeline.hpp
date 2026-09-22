  #pragma once

#include "dataset.hpp"

#include <atomic>
#include <cstddef>
  #include <string>
  #include <vector>

  class MonitorPotencia;

  struct ConfiguracaoPipeline {
      int runners = 2;
      int nucleos_cpu = 4;
      int workers_pre = 2;
      int workers_pos = 1;
      int slots_por_runner = 2;
      int warmup = 10;
      std::size_t inferencias = 0; // 0 = uma por amostra
      bool fixar_afinidade = false;
  };

  struct TemposImagem {
      std::size_t trabalho = 0;
      std::size_t indice_amostra = 0;

      double espera_slot_ms = 0;
      double leitura_ms = 0;
      double preprocess_ms = 0;
      double espera_runner_ms = 0;
      double sync_entrada_ms = 0;
      double inferencia_ms = 0;
      double sync_saida_ms = 0;
      double espera_pos_ms = 0;
      double postprocess_ms = 0;
      double latencia_total_ms = 0;
  };

  struct ResultadoExecucao {
      std::string modo;
      ConfiguracaoPipeline configuracao;
      std::size_t concluidas = 0;
      double duracao_s = 0;
      double throughput_fps = 0;
      std::vector<TemposImagem> imagens;
  };

  ResultadoExecucao executar_model_only(
      const std::string& caminho_xmodel,
      const std::vector<Amostra>& amostras,
      const ConfiguracaoPipeline& configuracao,
      MonitorPotencia* potencia = nullptr,
      std::atomic<std::size_t>* progresso = nullptr
  );

  ResultadoExecucao executar_end_to_end(
      const std::string& caminho_xmodel,
      const std::vector<Amostra>& amostras,
      const ConfiguracaoPipeline& configuracao,
      MonitorPotencia* potencia = nullptr,
      std::atomic<std::size_t>* progresso = nullptr
  );

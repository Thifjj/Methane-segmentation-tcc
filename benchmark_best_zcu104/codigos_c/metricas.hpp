  #pragma once

  #include "dataset.hpp"

  #include <array>
  #include <cstdint>
  #include <string>
  #include <vector>
  #include <cstddef>

  struct Contagens {
      std::uint64_t tp = 0;
      std::uint64_t fp = 0;
      std::uint64_t fn = 0;
      std::uint64_t tn = 0;
  };

  struct Metricas {
      double precision = 0;
      double recall = 0;
      double f1 = 0;
      double iou = 0;
      double fpr = 0;
      double acuracia = 0;
  };

  struct ResultadoImagem {
      std::string id;
      std::string dificuldade;
      Contagens contagens;
      Metricas metricas;
  };

  struct ResumoMetricas {
      std::size_t imagens = 0;
      Contagens global;
      Contagens forte;
      Contagens fraca;
      Contagens sem_pluma;
      Metricas metricas_globais;
      Metricas metricas_fortes;
      Metricas metricas_fracas;
      double auprc = 0;
      double fpr_sem_pluma = 0;
      std::uint64_t fp_tiles = 0;
      std::uint64_t tn_tiles = 0;
      double fpr_tile = 0;
      double fpr_tile_tabela = 0;
  };

  Metricas calcular_metricas(const Contagens& contagens);

  class AcumuladorMetricas {
  public:
      ResultadoImagem adicionar(
          const Amostra& amostra,
          const std::vector<std::uint8_t>& mascara,
          const cv::Mat& label,
          const void* logits,
          bool saida_float,
          float escala_saida
      );

      ResumoMetricas resumo() const;

  private:
      std::size_t imagens_ = 0;
      Contagens global_;
      Contagens forte_;
      Contagens fraca_;
      Contagens sem_pluma_;
      std::array<std::uint64_t, 256> positivos_{};
      std::array<std::uint64_t, 256> negativos_{};
      std::uint64_t fp_tiles_ = 0;
      std::uint64_t tn_tiles_ = 0;
  };

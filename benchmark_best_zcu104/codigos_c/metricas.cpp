#include "metricas.hpp"
  #include "postprocess.hpp"

  #include <algorithm>
  #include <cmath>
  #include <numeric>
  #include <stdexcept>

  namespace {

  void somar(Contagens& destino, const Contagens& origem) {
      destino.tp += origem.tp;
      destino.fp += origem.fp;
      destino.fn += origem.fn;
      destino.tn += origem.tn;
  }

  int indice_score(
      const void* logits,
      bool saida_float,
      float escala_saida,
      std::size_t pixel
  ) {
      if (!saida_float) {
          const auto* valores = static_cast<const std::int8_t*>(logits);
          return static_cast<int>(valores[pixel]) + 128;
      }

      const auto* valores = static_cast<const float*>(logits);
      float score = valores[pixel];
      if (!std::isfinite(score)) {
          throw std::runtime_error("Logit FLOAT32 não finito");
      }

      float quantizado = std::clamp(score / escala_saida, -128.0f, 127.0f);
      return static_cast<int>(std::lrint(quantizado)) + 128;
  }

  } // namespace

  Metricas calcular_metricas(const Contagens& c) {
      Metricas m;

      if (c.tp + c.fp > 0) {
          m.precision = static_cast<double>(c.tp) / (c.tp + c.fp);
      }
      if (c.tp + c.fn > 0) {
          m.recall = static_cast<double>(c.tp) / (c.tp + c.fn);
      }
      if (2 * c.tp + c.fp + c.fn > 0) {
          m.f1 = static_cast<double>(2 * c.tp) /
                 (2 * c.tp + c.fp + c.fn);
      }
      if (c.tp + c.fp + c.fn > 0) {
          m.iou = static_cast<double>(c.tp) /
                  (c.tp + c.fp + c.fn);
      }
      if (c.fp + c.tn > 0) {
          m.fpr = static_cast<double>(c.fp) / (c.fp + c.tn);
      }

      const auto total = c.tp + c.fp + c.fn + c.tn;
      if (total > 0) {
          m.acuracia = static_cast<double>(c.tp + c.tn) / total;
      }
      return m;
  }

  ResultadoImagem AcumuladorMetricas::adicionar(
      const Amostra& amostra,
      const std::vector<std::uint8_t>& mascara,
      const cv::Mat& label,
      const void* logits,
      bool saida_float,
      float escala_saida
  ) {
      if (mascara.size() != PIXELS_SAIDA ||
          label.rows != 512 || label.cols != 512 ||
          label.type() != CV_32FC1 || logits == nullptr) {
          throw std::runtime_error("Entrada inválida para calcular métricas");
      }
      if (saida_float &&
          (!std::isfinite(escala_saida) || escala_saida <= 0.0f)) {
          throw std::runtime_error("Escala de saída inválida");
      }

      Contagens c;

      for (int y = 0; y < 512; ++y) {
          const float* verdade = label.ptr<float>(y);

        for (int x = 0; x < 512; ++x) {
            if (!std::isfinite(verdade[x])) {
                throw std::runtime_error("Label com valor nao finito");
            }
            const std::size_t pixel = static_cast<std::size_t>(y) * 512 + x;
              const bool predicao = mascara[pixel] != 0;
              const bool real = verdade[x] != 0.0f;

              if (predicao && real) ++c.tp;
              else if (predicao) ++c.fp;
              else if (real) ++c.fn;
              else ++c.tn;

              const int bin = indice_score(
                  logits, saida_float, escala_saida, pixel
              );
              if (real) ++positivos_[bin];
              else ++negativos_[bin];
          }
      }

      ResultadoImagem resultado;
      resultado.id = amostra.id;
      resultado.contagens = c;
      resultado.metricas = calcular_metricas(c);

      somar(global_, c);
      ++imagens_;

      const std::uint64_t pixels_pluma = c.tp + c.fn;
      if (pixels_pluma == 0) {
          resultado.dificuldade = "sem_pluma";
          somar(sem_pluma_, c);
      } else if (amostra.qplume >= 1000.0 || pixels_pluma > 1000) {
          resultado.dificuldade = "forte";
          somar(forte_, c);
      } else {
          resultado.dificuldade = "fraca";
          somar(fraca_, c);
      }

      return resultado;
  }

  ResumoMetricas AcumuladorMetricas::resumo() const {
      ResumoMetricas r;
      r.imagens = imagens_;
      r.global = global_;
      r.forte = forte_;
      r.fraca = fraca_;
      r.sem_pluma = sem_pluma_;
      r.metricas_globais = calcular_metricas(global_);
      r.metricas_fortes = calcular_metricas(forte_);
      r.metricas_fracas = calcular_metricas(fraca_);
      r.fpr_sem_pluma = calcular_metricas(sem_pluma_).fpr;

      const std::uint64_t total_positivos =
          std::accumulate(positivos_.begin(), positivos_.end(), std::uint64_t{0});

      if (total_positivos == 0) return r;

      std::uint64_t tp = 0;
      std::uint64_t fp = 0;

      for (int bin = 255; bin >= 0; --bin) {
          const auto novos_tp = positivos_[bin];
          tp += novos_tp;
          fp += negativos_[bin];

          if (novos_tp > 0) {
              const double precision =
                  static_cast<double>(tp) / (tp + fp);
              r.auprc += precision *
                         static_cast<double>(novos_tp) / total_positivos;
          }
      }

      return r;
  }

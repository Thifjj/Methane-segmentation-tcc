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
      std::uint64_t pixels_preditos = 0;

      for (int y = 0; y < 512; ++y) {
          const float* verdade = label.ptr<float>(y);

        for (int x = 0; x < 512; ++x) {
            if (!std::isfinite(verdade[x])) {
                throw std::runtime_error("Label com valor nao finito");
            }
            const std::size_t pixel = static_cast<std::size_t>(y) * 512 + x;
              const bool predicao = mascara[pixel] != 0;
              const bool real = verdade[x] != 0.0f;
              pixels_preditos += predicao;

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

      if (!amostra.has_plume) {
          resultado.dificuldade = "sem_pluma";
          somar(sem_pluma_, c);
          if (pixels_preditos > 10) ++fp_tiles_;
          else ++tn_tiles_;
      } else if (amostra.qplume > 1000.0) {
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
      r.fp_tiles = fp_tiles_;
      r.tn_tiles = tn_tiles_;
      if (fp_tiles_ + tn_tiles_ > 0)
          r.fpr_tile = static_cast<double>(fp_tiles_) / (fp_tiles_ + tn_tiles_);
      if (imagens_ > 0)
          r.fpr_tile_tabela = static_cast<double>(fp_tiles_) / imagens_;

      const std::uint64_t total_positivos =
          std::accumulate(positivos_.begin(), positivos_.end(), std::uint64_t{0});

      // sklearn.precision_recall_curve + auc retorna 0.5 sem pixels positivos.
      if (total_positivos == 0) {
          if (imagens_ > 0) r.auprc = 0.5;
          return r;
      }

      std::uint64_t tp = total_positivos;
      std::uint64_t fp =
          std::accumulate(negativos_.begin(), negativos_.end(), std::uint64_t{0});
      for (std::size_t bin = 0; bin < positivos_.size(); ++bin) {
          const double precision_antes = static_cast<double>(tp) / (tp + fp);
          tp -= positivos_[bin];
          fp -= negativos_[bin];
          const double precision_depois = tp + fp > 0
              ? static_cast<double>(tp) / (tp + fp) : 1.0;
          r.auprc += static_cast<double>(positivos_[bin]) / total_positivos *
                     (precision_antes + precision_depois) / 2.0;
      }

      return r;
  }

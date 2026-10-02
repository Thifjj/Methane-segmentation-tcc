#include "metricas.hpp"
  #include "postprocess.hpp"

  #include <algorithm>
  #include <cmath>
  #include <numeric>
#include <limits>
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

      const float quantizado = score / escala_saida;
      if (!std::isfinite(quantizado) || quantizado < -128.0f || quantizado > 127.0f ||
          std::abs(quantizado - std::round(quantizado)) > 1e-4f)
          throw std::runtime_error("Saida FLOAT32 nao corresponde a grade INT8 do XModel");
      return static_cast<int>(std::lrint(quantizado)) + 128;
  }

  // Probabilidades FLOAT32 podem empatar por saturacao do sigmoid.
  double area_pr(const std::array<std::uint64_t, 256>& positivos,
                 const std::array<std::uint64_t, 256>& negativos,
                 float escala, bool average_precision) {
      const auto total = std::accumulate(positivos.begin(), positivos.end(), std::uint64_t{0});
      if (!total) return average_precision ? 0.0 : 0.5;
      std::uint64_t tp = 0, fp = 0;
      double area = 0, precision_anterior = 1;
      for (int bin = 255; bin >= 0;) {
          const float score = 1.0f / (1.0f + std::exp(-(bin - 128) * escala));
          std::uint64_t p = 0, n = 0;
          do {
              p += positivos[bin]; n += negativos[bin]; --bin;
          } while (bin >= 0 &&
                   1.0f / (1.0f + std::exp(-(bin - 128) * escala)) == score);
          if (p + n == 0) continue;
          tp += p; fp += n;
          const double precision = static_cast<double>(tp) / (tp + fp);
          area += static_cast<double>(p) / total *
                  (average_precision ? precision : (precision_anterior + precision) / 2);
          precision_anterior = precision;
      }
      return area;
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
      if (!std::isfinite(escala_saida) || escala_saida <= 0.0f) {
          throw std::runtime_error("Escala de saída inválida");
      }

      if (escala_saida_ != 0 && escala_saida_ != escala_saida)
          throw std::runtime_error("Escala de saida mudou durante a validacao");
      escala_saida_ = escala_saida;
      std::array<std::uint64_t, 256> positivos_imagem{}, negativos_imagem{};
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
              if (real) { ++positivos_[bin]; ++positivos_imagem[bin]; }
              else { ++negativos_[bin]; ++negativos_imagem[bin]; }
          }
      }

      ResultadoImagem resultado;
      resultado.id = amostra.id;
      resultado.contagens = c;
      resultado.metricas = calcular_metricas(c);
      resultado.positive = c.tp + c.fn > 0;
      resultado.difficulty_csv = amostra.difficulty;
      resultado.average_precision = area_pr(positivos_imagem, negativos_imagem, escala_saida, true);
      if (resultado.positive) { ++imagens_positivas_; soma_ap_ += resultado.average_precision; }

      somar(global_, c);
      ++imagens_;

      if (protocolo_oficial_ ? !resultado.positive : !amostra.has_plume) {
          resultado.dificuldade = "sem_pluma";
          somar(sem_pluma_, c);
          if (pixels_preditos > 10 * PIXELS_SAIDA / (64 * 64)) ++fp_tiles_;
          else ++tn_tiles_;
      } else if (protocolo_oficial_ ? amostra.difficulty == "easy" : amostra.qplume > 1000.0) {
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

      r.protocolo_oficial = protocolo_oficial_;
      r.imagens_positivas = imagens_positivas_;
      r.auprc = protocolo_oficial_
          ? (imagens_positivas_ ? soma_ap_ / imagens_positivas_ : std::numeric_limits<double>::quiet_NaN())
          : (imagens_ ? area_pr(positivos_, negativos_, escala_saida_, false) : 0.0);
      if (protocolo_oficial_) {
          // Mesmos denominadores de evaluate_quantized.summarize.
          const auto ajustar = [](const Contagens& c, Metricas& m) {
              m.f1 = 2.0 * c.tp / (2.0 * c.tp + c.fp + c.fn + 1e-6);
              m.iou = static_cast<double>(c.tp) / (c.tp + c.fp + c.fn + 1e-6);
          };
          ajustar(global_, r.metricas_globais);
          ajustar(forte_, r.metricas_fortes);
          ajustar(fraca_, r.metricas_fracas);
          r.fpr_sem_pluma = static_cast<double>(sem_pluma_.fp) /
                           (sem_pluma_.fp + sem_pluma_.tn + 1e-6);
      }

      return r;
  }

#include "postprocess.hpp"

  #include <stdexcept>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>

  void posprocessar(
      const void* dados_saida,
      std::size_t bytes_saida,
      bool saida_float,
      std::vector<std::uint8_t>& mascara
  ) {
      const std::size_t esperado =
          PIXELS_SAIDA * (saida_float ? sizeof(float) : sizeof(std::int8_t));

      if (dados_saida == nullptr || bytes_saida != esperado) {
          throw std::runtime_error("Buffer de saída inválido");
      }

      mascara.resize(PIXELS_SAIDA);

      if (saida_float) {
          const auto* logits = static_cast<const float*>(dados_saida);
          for (std::size_t i = 0; i < PIXELS_SAIDA; ++i) {
              if (!std::isfinite(logits[i])) throw std::runtime_error("Logit FLOAT32 nao finito");
              mascara[i] = logits[i] > 0.0f;
          }
      } else {
          const auto* logits = static_cast<const std::int8_t*>(dados_saida);
          for (std::size_t i = 0; i < PIXELS_SAIDA; ++i) {
              mascara[i] = logits[i] > 0;
          }
      }
  }

void reconstruir_patches(const void* origem, void* destino,
                         int tamanho_patch, std::size_t bytes_por_pixel) {
    if (!origem || !destino || (tamanho_patch != 128 && tamanho_patch != 512) ||
        (bytes_por_pixel != 1 && bytes_por_pixel != sizeof(float)))
        throw std::runtime_error("Buffers ou tamanho de patch invalidos");
    const auto* entrada = static_cast<const std::uint8_t*>(origem);
    auto* saida = static_cast<std::uint8_t*>(destino);
    for (int y = 0; y < 512; ++y)
        for (int x = 0; x < 512; x += tamanho_patch) {
            const std::size_t patch = (y / tamanho_patch) * (512 / tamanho_patch) + x / tamanho_patch;
            const std::size_t offset = patch * tamanho_patch * tamanho_patch +
                                       (y % tamanho_patch) * tamanho_patch;
            std::memcpy(saida + (y * 512 + x) * bytes_por_pixel,
                        entrada + offset * bytes_por_pixel, tamanho_patch * bytes_por_pixel);
        }
}

void abrir_mascara(std::vector<std::uint8_t>& mascara) {
    if (mascara.size() != PIXELS_SAIDA) throw std::runtime_error("Mascara invalida");
    cv::Mat imagem(512, 512, CV_8UC1, mascara.data());
    // Bordas neutras: equivalentes ao border_type=geodesic do Kornia.
    cv::morphologyEx(imagem, imagem, cv::MORPH_OPEN,
                    cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
}

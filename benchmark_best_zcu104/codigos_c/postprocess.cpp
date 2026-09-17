#include "postprocess.hpp"

  #include <stdexcept>

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
              mascara[i] = logits[i] >= 0.0f;
          }
      } else {
          const auto* logits = static_cast<const std::int8_t*>(dados_saida);
          for (std::size_t i = 0; i < PIXELS_SAIDA; ++i) {
              mascara[i] = logits[i] >= 0;
          }
      }
  }
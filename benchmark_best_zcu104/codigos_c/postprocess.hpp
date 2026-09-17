#pragma once

  #include <cstddef>
  #include <cstdint>
  #include <vector>

  constexpr std::size_t PIXELS_SAIDA = 512 * 512;

  void posprocessar(
      const void* dados_saida,
      std::size_t bytes_saida,
      bool saida_float,
      std::vector<std::uint8_t>& mascara
  );
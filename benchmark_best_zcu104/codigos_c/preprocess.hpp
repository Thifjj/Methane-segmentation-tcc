#pragma once

  #include "dataset.hpp"

  #include <cstddef>
  #include <cstdint>

  constexpr std::size_t TAMANHO_ENTRADA = 512 * 512 * 4;

  // escala_entrada vem do tensor do XModel: 2^fix_point.
  void preprocessar(
      const CanaisEntrada& canais,
      std::int8_t* destino,
      std::size_t capacidade_destino,
      float escala_entrada
  );
#pragma once

#include "metricas.hpp"

#include <atomic>
#include <string>
#include <vector>

struct ResultadoValidacao {
    std::vector<ResultadoImagem> imagens;
    ResumoMetricas resumo;
};

ResultadoValidacao validar_modelo(
    const std::string& caminho_xmodel,
    const std::vector<Amostra>& amostras,
    int warmup,
    std::atomic<std::size_t>* progresso = nullptr
);

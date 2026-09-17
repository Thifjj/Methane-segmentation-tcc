#pragma once

#include "metricas.hpp"
#include "pipeline.hpp"
#include "power.hpp"

#include <filesystem>
#include <string>
#include <vector>

void salvar_desempenho(
    const std::filesystem::path& pasta,
    const std::string& modelo,
    const std::string& run_id,
    const ResultadoExecucao& execucao,
    const std::vector<MedidaPotencia>& potencia,
    bool potencia_solicitada
);

void salvar_metricas(
    const std::filesystem::path& pasta,
    const std::string& modelo,
    const std::vector<ResultadoImagem>& imagens,
    const ResumoMetricas& resumo
);

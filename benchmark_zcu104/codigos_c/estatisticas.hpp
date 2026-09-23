#pragma once

#include "pipeline.hpp"

#include <cstddef>
#include <vector>

struct Estatisticas {
    std::size_t amostras = 0;
    double media = 0;
    double mediana = 0;
    double minimo = 0;
    double maximo = 0;
    double p95 = 0;
    double p99 = 0;
    double desvio_padrao = 0;
};

struct ResumoDesempenho {
    Estatisticas latencia_total;
    Estatisticas espera_slot;
    Estatisticas leitura;
    Estatisticas preprocess;
    Estatisticas espera_runner;
    Estatisticas sync_entrada;
    Estatisticas inferencia;
    Estatisticas sync_saida;
    Estatisticas espera_pos;
    Estatisticas postprocess;
};

Estatisticas resumir_valores(std::vector<double> valores);
ResumoDesempenho resumir_execucao(const ResultadoExecucao& execucao);

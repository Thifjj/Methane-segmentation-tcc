#include "estatisticas.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

Estatisticas resumir_valores(std::vector<double> valores) {
    Estatisticas r;
    if (valores.empty()) return r;
    std::sort(valores.begin(), valores.end());
    r.amostras = valores.size();
    r.minimo = valores.front();
    r.maximo = valores.back();
    r.media = std::accumulate(valores.begin(), valores.end(), 0.0) / valores.size();

    const auto percentil = [&](double p) {
        const double pos = p * (valores.size() - 1);
        const auto i = static_cast<std::size_t>(pos);
        const double fracao = pos - i;
        return valores[i] * (1.0 - fracao) +
               valores[std::min(i + 1, valores.size() - 1)] * fracao;
    };
    r.mediana = percentil(0.5);
    r.p95 = percentil(0.95);
    r.p99 = percentil(0.99);
    double soma_quadrados = 0;
    for (double valor : valores) soma_quadrados += (valor - r.media) * (valor - r.media);
    r.desvio_padrao = std::sqrt(soma_quadrados / valores.size());
    return r;
}

ResumoDesempenho resumir_execucao(const ResultadoExecucao& execucao) {
    ResumoDesempenho r;
    const auto extrair = [&](double TemposImagem::*campo) {
        std::vector<double> valores;
        valores.reserve(execucao.imagens.size());
        for (const auto& imagem : execucao.imagens) valores.push_back(imagem.*campo);
        return resumir_valores(std::move(valores));
    };
    r.latencia_total = extrair(&TemposImagem::latencia_total_ms);
    r.espera_slot = extrair(&TemposImagem::espera_slot_ms);
    r.leitura = extrair(&TemposImagem::leitura_ms);
    r.preprocess = extrair(&TemposImagem::preprocess_ms);
    r.espera_runner = extrair(&TemposImagem::espera_runner_ms);
    r.sync_entrada = extrair(&TemposImagem::sync_entrada_ms);
    r.inferencia = extrair(&TemposImagem::inferencia_ms);
    r.sync_saida = extrair(&TemposImagem::sync_saida_ms);
    r.espera_pos = extrair(&TemposImagem::espera_pos_ms);
    r.postprocess = extrair(&TemposImagem::postprocess_ms);
    return r;
}

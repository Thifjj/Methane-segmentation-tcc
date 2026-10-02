#include "dataset.hpp"
#include "estatisticas.hpp"
#include "metricas.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "progresso.hpp"
#include "power.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

static void exigir(bool condicao) {
    if (!condicao) throw std::runtime_error("Self-test falhou");
}

int main() {
    std::vector<std::int8_t> logits(PIXELS_SAIDA, -1);
    logits[0] = 1;
    std::vector<std::uint8_t> mascara;
    posprocessar(logits.data(), logits.size(), false, mascara);
    exigir(mascara[0] == 1 && mascara[1] == 0);

    cv::Mat label(512, 512, CV_32FC1, cv::Scalar(0));
    label.at<float>(0, 0) = 1;
    Amostra amostra;
    amostra.id = "teste";
    amostra.has_plume = true;
    amostra.qplume = 1000.0;
    AcumuladorMetricas acumulador;
    const auto imagem = acumulador.adicionar(
        amostra, mascara, label, logits.data(), false, 1.0f);
    const auto resumo = acumulador.resumo();
    exigir(imagem.contagens.tp == 1 && imagem.contagens.fp == 0);
    exigir(resumo.metricas_globais.f1 == 1.0);
    exigir(resumo.metricas_fracas.f1 == 1.0);
    exigir(resumo.auprc == 1.0);

    amostra.qplume = 1000.1;
    AcumuladorMetricas acumulador_forte;
    acumulador_forte.adicionar(
        amostra, mascara, label, logits.data(), false, 1.0f);
    exigir(acumulador_forte.resumo().metricas_fortes.f1 == 1.0);

    amostra.has_plume = false;
    AcumuladorMetricas acumulador_sem_pluma;
    const auto sem_pluma = acumulador_sem_pluma.adicionar(
        amostra, mascara, label, logits.data(), false, 1.0f);
    exigir(sem_pluma.dificuldade == "sem_pluma");
    exigir(acumulador_sem_pluma.resumo().metricas_fortes.f1 == 0.0);

    std::fill(mascara.begin(), mascara.end(), 0);
    std::fill(mascara.begin(), mascara.begin() + 640, 1);
    AcumuladorMetricas limite_tile;
    limite_tile.adicionar(amostra, mascara, cv::Mat(512, 512, CV_32FC1, cv::Scalar(0)),
                        logits.data(), false, 1.0f);
    mascara[640] = 1;
    limite_tile.adicionar(amostra, mascara, cv::Mat(512, 512, CV_32FC1, cv::Scalar(0)),
                        logits.data(), false, 1.0f);
    exigir(limite_tile.resumo().fp_tiles == 1 && limite_tile.resumo().tn_tiles == 1);
    exigir(limite_tile.resumo().fpr_tile == 0.5);

    const auto tempos = resumir_valores({1.0, 2.0, 3.0});
    exigir(tempos.media == 2.0 && tempos.mediana == 2.0);
    exigir(std::abs(tempos.p95 - 2.9) < 1e-12);

    std::ostringstream saida;
    {
        Progresso progresso("teste", 3, saida);
        progresso.contador()->store(3);
    }
    exigir(saida.str().find("teste: 3/3") != std::string::npos);

    CanaisEntrada canais;
    const std::array<float, 4> divisores{1750.0f, 60.0f, 60.0f, 60.0f};
    for (std::size_t c = 0; c < 4; ++c) {
        canais[c].create(512, 512, CV_32FC1);
        for (int y = 0; y < 512; ++y) {
            float* linha = canais[c].ptr<float>(y);
            for (int x = 0; x < 512; ++x) {
                linha[x] = ((x * 37 + y * 11 + static_cast<int>(c) * 53) % 5000
                            - 250) * divisores[c] / 1000.0f;
                if (x % 17 == 0)
                    linha[x] = (static_cast<float>(x % 64) + 0.5f) *
                               divisores[c] / 32.0f;
            }
        }
    }
    std::vector<std::int8_t> entrada(TAMANHO_ENTRADA);
    preprocessar(canais, entrada.data(), entrada.size(), 32.0f);
    for (int y = 0; y < 512; ++y)
        for (int x = 0; x < 512; ++x)
            for (std::size_t c = 0; c < 4; ++c) {
                const float normalizado = std::clamp(
                    canais[c].at<float>(y, x) / divisores[c], 0.0f, 2.0f);
                const auto esperado = static_cast<std::int8_t>(
                    std::clamp(static_cast<long>(std::floor(normalizado * 32.0f + 0.5f)), -128L, 127L));
                exigir(entrada[(static_cast<std::size_t>(y) * 512 + x) * 4 + c] ==
                       esperado);
            }
    exigir(entrada[0] == 1); // Empate 0.5: Vitis/DPU arredonda para cima.
    std::vector<std::int8_t> patches(TAMANHO_ENTRADA);
    preprocessar(canais, patches.data(), patches.size(), 32.0f, 128);
    // Reconstrucao canal a canal; o layout de entrada e NHWC.
    std::vector<float> saida_patches(PIXELS_SAIDA), saida_reconstruida(PIXELS_SAIDA);
    std::vector<std::int8_t> saida_patches_int8(PIXELS_SAIDA), saida_reconstruida_int8(PIXELS_SAIDA);
    for (int y = 0; y < 512; ++y)
        for (int x = 0; x < 512; ++x) {
            const std::size_t pixel = y * 512 + x;
            const std::size_t patch_pixel = ((y / 128) * 4 + x / 128) * 128 * 128 +
                                            (y % 128) * 128 + x % 128;
            for (std::size_t c = 0; c < 4; ++c)
                exigir(patches[patch_pixel * 4 + c] == entrada[pixel * 4 + c]);
            saida_patches[patch_pixel] = static_cast<float>(pixel);
            saida_patches_int8[patch_pixel] = static_cast<std::int8_t>(pixel % 127);
        }
    reconstruir_patches(saida_patches.data(), saida_reconstruida.data(), 128, sizeof(float));
    reconstruir_patches(saida_patches_int8.data(), saida_reconstruida_int8.data(), 128, 1);
    for (std::size_t i = 0; i < PIXELS_SAIDA; ++i) {
        exigir(saida_reconstruida[i] == static_cast<float>(i));
        exigir(saida_reconstruida_int8[i] == static_cast<std::int8_t>(i % 127));
    }

    // Cruz atravessa a fronteira entre patches; abertura so depois de recompor.
    std::fill(mascara.begin(), mascara.end(), 0);
    for (auto i : {128*512+128, 127*512+128, 129*512+128, 128*512+127, 128*512+129})
        mascara[i] = 1;
    mascara[0] = 1; // Pixel isolado na borda.
    abrir_mascara(mascara);
    exigir(std::count(mascara.begin(), mascara.end(), 1) == 5 && !mascara[0]);

    // Scores 17 e 20 empatam em sigmoid FLOAT32=1: AP=1/2, AUC PR=3/4.
    std::fill(logits.begin(), logits.end(), -128);
    logits[0] = 40; logits[1] = 34;
    posprocessar(logits.data(), logits.size(), false, mascara);
    amostra.has_plume = false; amostra.qplume = 0; amostra.difficulty = "easy";
    AcumuladorMetricas oficial(true), legado;
    auto positivo = oficial.adicionar(amostra, mascara, label, logits.data(), false, 0.5f);
    legado.adicionar(amostra, mascara, label, logits.data(), false, 0.5f);
    exigir(positivo.dificuldade == "forte" && positivo.positive);
    exigir(std::abs(positivo.average_precision - 0.5) < 1e-12);
    exigir(std::abs(legado.resumo().auprc - 0.75) < 1e-12);
    oficial.adicionar(amostra, mascara, cv::Mat(512, 512, CV_32FC1, cv::Scalar(0)),
                      logits.data(), false, 0.5f);
    exigir(oficial.resumo().imagens_positivas == 1 && oficial.resumo().auprc == 0.5);
    exigir(oficial.resumo().sem_pluma.fp == 2);
    std::vector<float> logits_float(PIXELS_SAIDA);
    for (std::size_t i = 0; i < PIXELS_SAIDA; ++i) logits_float[i] = logits[i] * 0.5f;
    AcumuladorMetricas oficial_float(true);
    const auto float_imagem = oficial_float.adicionar(amostra, mascara, label,
                                                      logits_float.data(), true, 0.5f);
    exigir(float_imagem.average_precision == positivo.average_precision);
    AcumuladorMetricas sem_positivos(true);
    sem_positivos.adicionar(amostra, mascara, cv::Mat(512, 512, CV_32FC1, cv::Scalar(0)),
                           logits.data(), false, 0.5f);
    exigir(std::isnan(sem_positivos.resumo().auprc));
    logits_float[0] = 0.1f;
    bool rejeitou_grade = false;
    try {
        AcumuladorMetricas fora_da_grade;
        fora_da_grade.adicionar(amostra, mascara, label, logits_float.data(), true, 0.5f);
    } catch (const std::runtime_error&) { rejeitou_grade = true; }
    exigir(rejeitou_grade);
    exigir(std::abs(integrar_potencia({{0, 0}, {1, 2}, {3, 2}}, 0.5, 2.5) - 3.75) < 1e-12);
    exigir(integrar_potencia({{1, 2}}, 0, 3) == 6);
    exigir(integrar_potencia({}, 0, 3) == 0);
    MonitorPotencia monitor(1);
    monitor.iniciar(); monitor.parar(); // Pode funcionar sem sensores no computador.

    canais[0].at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    bool rejeitou_nan = false;
    try {
        preprocessar(canais, entrada.data(), entrada.size(), 32.0f);
    } catch (const std::runtime_error&) {
        rejeitou_nan = true;
    }
    exigir(rejeitou_nan);

    std::cout << "Self-test OK\n";
}

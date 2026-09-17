#include "dataset.hpp"
#include "estatisticas.hpp"
#include "metricas.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "progresso.hpp"

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
    AcumuladorMetricas acumulador;
    const auto imagem = acumulador.adicionar(
        amostra, mascara, label, logits.data(), false, 1.0f);
    const auto resumo = acumulador.resumo();
    exigir(imagem.contagens.tp == 1 && imagem.contagens.fp == 0);
    exigir(resumo.metricas_globais.f1 == 1.0);
    exigir(resumo.metricas_fracas.f1 == 1.0);
    exigir(resumo.auprc == 1.0);

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
                    std::clamp(std::lrint(normalizado * 32.0f), -128L, 127L));
                exigir(entrada[(static_cast<std::size_t>(y) * 512 + x) * 4 + c] ==
                       esperado);
            }
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

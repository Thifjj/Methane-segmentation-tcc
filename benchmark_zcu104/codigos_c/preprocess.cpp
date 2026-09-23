#include "preprocess.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace {

#ifndef __aarch64__
std::int8_t quantizar(float valor, float divisor, float escala) {
    if (!std::isfinite(valor)) {
        throw std::runtime_error("Canal com valor nao finito");
    }
    float normalizado = std::clamp(valor / divisor, 0.0f, 2.0f);
    long quantizado = std::lrint(normalizado * escala);
    return static_cast<std::int8_t>(std::clamp(quantizado, -128L, 127L));
}
#endif

#ifdef __aarch64__
int8x8_t quantizar8(const float* origem, float divisor, float escala) {
    const auto divisor_v = vdupq_n_f32(divisor);
    const auto zero = vdupq_n_f32(0.0f);
    const auto dois = vdupq_n_f32(2.0f);
    const auto finito = vdupq_n_f32(FLT_MAX);
    const auto a = vld1q_f32(origem);
    const auto b = vld1q_f32(origem + 4);
    if (vminvq_u32(vcleq_f32(vabsq_f32(a), finito)) == 0 ||
        vminvq_u32(vcleq_f32(vabsq_f32(b), finito)) == 0)
        throw std::runtime_error("Canal com valor nao finito");

    const auto qa = vcvtnq_s32_f32(vmulq_n_f32(
        vmaxq_f32(zero, vminq_f32(vdivq_f32(a, divisor_v), dois)), escala));
    const auto qb = vcvtnq_s32_f32(vmulq_n_f32(
        vmaxq_f32(zero, vminq_f32(vdivq_f32(b, divisor_v), dois)), escala));
    return vqmovn_s16(vcombine_s16(vqmovn_s32(qa), vqmovn_s32(qb)));
}
#endif

} // namespace

void preprocessar(
    const CanaisEntrada& canais,
    std::int8_t* destino,
    std::size_t capacidade_destino,
    float escala_entrada
) {
    if (destino == nullptr || capacidade_destino < TAMANHO_ENTRADA) {
        throw std::runtime_error("Buffer de entrada INT8 inválido");
    }
    if (!std::isfinite(escala_entrada) || escala_entrada <= 0.0f) {
        throw std::runtime_error("Escala de entrada inválida");
    }

    for (const cv::Mat& canal : canais) {
        if (canal.rows != 512 || canal.cols != 512 || canal.type() != CV_32FC1) {
            throw std::runtime_error("Canal deve ser CV_32FC1 e 512x512");
        }
    }

    for (int y = 0; y < 512; ++y) {
        const float* mag1c = canais[0].ptr<float>(y);
        const float* vermelho = canais[1].ptr<float>(y);
        const float* verde = canais[2].ptr<float>(y);
        const float* azul = canais[3].ptr<float>(y);
        std::int8_t* linha = destino + static_cast<std::size_t>(y) * 512 * 4;

#ifdef __aarch64__
        for (int x = 0; x < 512; x += 8) {
            int8x8x4_t quantizados;
            quantizados.val[0] = quantizar8(mag1c + x, 1750.0f, escala_entrada);
            quantizados.val[1] = quantizar8(vermelho + x, 60.0f, escala_entrada);
            quantizados.val[2] = quantizar8(verde + x, 60.0f, escala_entrada);
            quantizados.val[3] = quantizar8(azul + x, 60.0f, escala_entrada);
            vst4_s8(linha + 4 * x, quantizados);
        }
#else
        for (int x = 0; x < 512; ++x) {
            linha[4 * x + 0] = quantizar(mag1c[x], 1750.0f, escala_entrada);
            linha[4 * x + 1] = quantizar(vermelho[x], 60.0f, escala_entrada);
            linha[4 * x + 2] = quantizar(verde[x], 60.0f, escala_entrada);
            linha[4 * x + 3] = quantizar(azul[x], 60.0f, escala_entrada);
        }
#endif
    }
}

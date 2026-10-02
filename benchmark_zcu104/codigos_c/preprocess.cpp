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
    long quantizado = static_cast<long>(std::floor(normalizado * escala + 0.5f));
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

    const auto qa = vcvtq_s32_f32(vaddq_f32(vdupq_n_f32(0.5f), vmulq_n_f32(
        vmaxq_f32(zero, vminq_f32(vdivq_f32(a, divisor_v), dois)), escala)));
    const auto qb = vcvtq_s32_f32(vaddq_f32(vdupq_n_f32(0.5f), vmulq_n_f32(
        vmaxq_f32(zero, vminq_f32(vdivq_f32(b, divisor_v), dois)), escala)));
    return vqmovn_s16(vcombine_s16(vqmovn_s32(qa), vqmovn_s32(qb)));
}
#endif

} // namespace

void preprocessar(
    const CanaisEntrada& canais,
    std::int8_t* destino,
    std::size_t capacidade_destino,
    float escala_entrada,
    int tamanho_patch
) {
    if (destino == nullptr || capacidade_destino < TAMANHO_ENTRADA) {
        throw std::runtime_error("Buffer de entrada INT8 inválido");
    }
    if (!std::isfinite(escala_entrada) || escala_entrada <= 0.0f) {
        throw std::runtime_error("Escala de entrada inválida");
    }

    if (tamanho_patch != 128 && tamanho_patch != 512)
        throw std::runtime_error("Patch deve ter 128 ou 512 pixels");

    for (const cv::Mat& canal : canais) {
        if (canal.rows != 512 || canal.cols != 512 || canal.type() != CV_32FC1) {
            throw std::runtime_error("Canal deve ser CV_32FC1 e 512x512");
        }
    }

    for (int y = 0; y < 512; ++y) {
        const float* mag1c = canais[0].ptr<float>(y);
        const float* banda1 = canais[1].ptr<float>(y);
        const float* banda2 = canais[2].ptr<float>(y);
        const float* banda3 = canais[3].ptr<float>(y);
        // Patches em ordem de linhas, NHWC dentro de cada patch (como tiled_logits).
        const auto offset = [&](int x) {
            const std::size_t patch = (y / tamanho_patch) * (512 / tamanho_patch) + x / tamanho_patch;
            return (patch * tamanho_patch * tamanho_patch +
                    (y % tamanho_patch) * tamanho_patch + x % tamanho_patch) * 4;
        };

#ifdef __aarch64__
        for (int x = 0; x < 512; x += 8) {
            int8x8x4_t quantizados;
            quantizados.val[0] = quantizar8(mag1c + x, 1750.0f, escala_entrada);
            quantizados.val[1] = quantizar8(banda1 + x, 60.0f, escala_entrada);
            quantizados.val[2] = quantizar8(banda2 + x, 60.0f, escala_entrada);
            quantizados.val[3] = quantizar8(banda3 + x, 60.0f, escala_entrada);
            vst4_s8(destino + offset(x), quantizados);
        }
#else
        for (int x = 0; x < 512; ++x) {
            destino[offset(x) + 0] = quantizar(mag1c[x], 1750.0f, escala_entrada);
            destino[offset(x) + 1] = quantizar(banda1[x], 60.0f, escala_entrada);
            destino[offset(x) + 2] = quantizar(banda2[x], 60.0f, escala_entrada);
            destino[offset(x) + 3] = quantizar(banda3[x], 60.0f, escala_entrada);
        }
#endif
    }
}

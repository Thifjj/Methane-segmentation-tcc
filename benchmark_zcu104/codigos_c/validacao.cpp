#include "validacao.hpp"

#include "postprocess.hpp"
#include "preprocess.hpp"
#include "xmodel_runner.hpp"

#include <stdexcept>

ResultadoValidacao validar_modelo(const std::string& caminho_xmodel,
                                  const std::vector<Amostra>& amostras,
                                  int warmup,
                                  std::atomic<std::size_t>* progresso) {
    if (amostras.empty() || warmup < 0)
        throw std::runtime_error("Amostras ou warmup invalidos para validacao");

    auto runner = carregar_xmodel(caminho_xmodel);
    auto& slot = *runner.slots.front();
    auto canais = carregar_canais(amostras.front(), runner.ordem_rgb);
    preprocessar(canais, slot.dados_entrada(), slot.bytes_entrada(),
                 runner.escala_entrada, runner.tamanho_patch);
    runner.sincronizar_entrada(slot);
    for (int i = 0; i < warmup; ++i) runner.inferir(slot);

    AcumuladorMetricas acumulador;
    AcumuladorMetricas oficial(true);
    ResultadoValidacao resultado;
    resultado.imagens.reserve(amostras.size());
    resultado.imagens_oficiais.reserve(amostras.size());
    std::vector<std::uint8_t> mascara(PIXELS_SAIDA);

    for (const auto& amostra : amostras) {
        canais = carregar_canais(amostra, runner.ordem_rgb);
        preprocessar(canais, slot.dados_entrada(), slot.bytes_entrada(),
                     runner.escala_entrada, runner.tamanho_patch);
        runner.sincronizar_entrada(slot);
        runner.inferir(slot);
        runner.sincronizar_saida(slot);
        posprocessar(slot.dados_saida(), slot.bytes_saida(),
                     runner.saida_float, mascara);
        const auto label = carregar_label(amostra);
        resultado.imagens.push_back(acumulador.adicionar(
            amostra, mascara, label, slot.dados_saida(),
            runner.saida_float, runner.escala_saida));
        abrir_mascara(mascara);
        resultado.imagens_oficiais.push_back(oficial.adicionar(
            amostra, mascara, label, slot.dados_saida(),
            runner.saida_float, runner.escala_saida));
        if (progresso) progresso->fetch_add(1, std::memory_order_relaxed);
    }
    resultado.resumo = acumulador.resumo();
    resultado.resumo_oficial = oficial.resumo();
    return resultado;
}

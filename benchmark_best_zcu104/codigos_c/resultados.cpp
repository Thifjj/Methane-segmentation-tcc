#include "resultados.hpp"

#include "estatisticas.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

std::string csv(const std::string& valor) {
    std::string resultado = "\"";
    for (char c : valor) {
        if (c == '"') resultado += '"';
        resultado += c;
    }
    return resultado + '"';
}

std::ofstream abrir(const fs::path& arquivo, const std::string& cabecalho) {
    const bool vazio = !fs::exists(arquivo) || fs::file_size(arquivo) == 0;
    std::ofstream saida(arquivo, std::ios::app);
    if (!saida) throw std::runtime_error("Nao foi possivel escrever: " + arquivo.string());
    saida << std::setprecision(12);
    if (vazio) saida << cabecalho << '\n';
    return saida;
}

void linha_grupo(std::ofstream& saida, const char* nome,
                 const Contagens& c, const Metricas& m) {
    saida << nome << ',' << c.tp << ',' << c.fp << ',' << c.fn << ',' << c.tn
          << ',' << m.precision << ',' << m.recall << ',' << m.f1 << ',' << m.iou
          << ',' << m.fpr << ',' << m.acuracia << '\n';
}

} // namespace

void salvar_desempenho(const fs::path& pasta, const std::string& modelo,
                       const std::string& run_id, const ResultadoExecucao& execucao,
                       const std::vector<MedidaPotencia>& potencia,
                       bool potencia_solicitada) {
    fs::create_directories(pasta);
    const auto& c = execucao.configuracao;
    const auto r = resumir_execucao(execucao);

    auto geral = abrir(pasta / "benchmark_geral.csv",
        "modelo,run_id,modo,runners,nucleos_cpu,workers_pre,workers_pos,slots_por_runner,"
        "fixar_afinidade,warmup,inferencias,duracao_s,throughput_fps,"
        "latencia_media_ms,latencia_mediana_ms,latencia_min_ms,latencia_max_ms,"
        "latencia_p95_ms,latencia_p99_ms,latencia_desvio_ms,inferencia_media_ms,"
        "leitura_media_ms,preprocess_media_ms,postprocess_media_ms");
    geral << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo << ','
          << c.runners << ',' << c.nucleos_cpu << ',' << c.workers_pre << ','
          << c.workers_pos << ',' << c.slots_por_runner << ','
          << c.fixar_afinidade << ',' << c.warmup << ',' << execucao.concluidas << ','
          << execucao.duracao_s << ',' << execucao.throughput_fps << ','
          << r.latencia_total.media << ',' << r.latencia_total.mediana << ','
          << r.latencia_total.minimo << ',' << r.latencia_total.maximo << ','
          << r.latencia_total.p95 << ',' << r.latencia_total.p99 << ','
          << r.latencia_total.desvio_padrao << ',' << r.inferencia.media << ','
          << r.leitura.media << ',' << r.preprocess.media << ','
          << r.postprocess.media << '\n';

    auto estagios = abrir(pasta / "benchmark_estagios.csv",
        "modelo,run_id,modo,estagio,amostras,media_ms,mediana_ms,minimo_ms,"
        "maximo_ms,p95_ms,p99_ms,desvio_padrao_ms");
    const auto gravar_estagio = [&](const char* nome, const Estatisticas& e) {
        estagios << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo << ','
                 << nome << ',' << e.amostras << ',' << e.media << ',' << e.mediana
                 << ',' << e.minimo << ',' << e.maximo << ',' << e.p95 << ','
                 << e.p99 << ',' << e.desvio_padrao << '\n';
    };
    gravar_estagio("latencia_total", r.latencia_total);
    gravar_estagio("espera_slot", r.espera_slot);
    gravar_estagio("leitura", r.leitura);
    gravar_estagio("preprocess", r.preprocess);
    gravar_estagio("espera_runner", r.espera_runner);
    gravar_estagio("sync_entrada", r.sync_entrada);
    gravar_estagio("inferencia", r.inferencia);
    gravar_estagio("sync_saida", r.sync_saida);
    gravar_estagio("espera_pos", r.espera_pos);
    gravar_estagio("postprocess", r.postprocess);

    auto amostras = abrir(pasta / "benchmark_samples.csv",
        "modelo,run_id,modo,trabalho,indice_amostra,espera_slot_ms,leitura_ms,"
        "preprocess_ms,espera_runner_ms,sync_entrada_ms,inferencia_ms,"
        "sync_saida_ms,espera_pos_ms,postprocess_ms,latencia_total_ms");
    for (const auto& t : execucao.imagens) {
        amostras << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo
                 << ',' << t.trabalho << ',' << t.indice_amostra << ','
                 << t.espera_slot_ms << ',' << t.leitura_ms << ','
                 << t.preprocess_ms << ',' << t.espera_runner_ms << ','
                 << t.sync_entrada_ms << ',' << t.inferencia_ms << ','
                 << t.sync_saida_ms << ',' << t.espera_pos_ms << ','
                 << t.postprocess_ms << ',' << t.latencia_total_ms << '\n';
    }

    auto energia = abrir(pasta / "benchmark_power_rails.csv",
        "modelo,run_id,modo,trilho,status,amostras,media_w,minima_w,maxima_w,energia_j");
    if (potencia.empty())
        energia << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo
                << (potencia_solicitada ? ",,indisponivel,0,,,,\n" :
                                         ",,desativada,0,,,,\n");
    for (const auto& p : potencia)
        energia << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo
                << ',' << csv(p.trilho) << ",ok," << p.amostras << ',' << p.media_w
                << ',' << p.minima_w << ',' << p.maxima_w << ',' << p.energia_j << '\n';
}

void salvar_metricas(const fs::path& pasta, const std::string& modelo,
                     const std::vector<ResultadoImagem>& imagens,
                     const ResumoMetricas& resumo) {
    if (imagens.size() != resumo.imagens)
        throw std::runtime_error("Quantidade de imagens e resumo nao coincidem");
    fs::create_directories(pasta);

    std::ofstream por_imagem(pasta / "metricas_por_imagem.csv");
    if (!por_imagem) throw std::runtime_error("Nao foi possivel salvar metricas por imagem");
    por_imagem << std::setprecision(12)
               << "modelo,id,dificuldade,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia\n";
    for (const auto& imagem : imagens) {
        const auto& c = imagem.contagens;
        const auto& m = imagem.metricas;
        por_imagem << csv(modelo) << ',' << csv(imagem.id) << ','
                   << imagem.dificuldade << ',' << c.tp << ',' << c.fp << ','
                   << c.fn << ',' << c.tn << ',' << m.precision << ',' << m.recall
                   << ',' << m.f1 << ',' << m.iou << ',' << m.fpr << ','
                   << m.acuracia << '\n';
    }

    std::ofstream global(pasta / "metricas_globais.csv");
    if (!global) throw std::runtime_error("Nao foi possivel salvar metricas globais");
    global << std::setprecision(12)
           << "modelo,imagens,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia,"
           << "f1_forte,f1_fraca,auprc,fpr_sem_pluma\n";
    const auto& c = resumo.global;
    const auto& m = resumo.metricas_globais;
    global << csv(modelo) << ',' << resumo.imagens << ',' << c.tp << ',' << c.fp
           << ',' << c.fn << ',' << c.tn << ',' << m.precision << ',' << m.recall
           << ',' << m.f1 << ',' << m.iou << ',' << m.fpr << ',' << m.acuracia
           << ',' << resumo.metricas_fortes.f1 << ',' << resumo.metricas_fracas.f1
           << ',' << resumo.auprc << ',' << resumo.fpr_sem_pluma << '\n';

    std::ofstream grupos(pasta / "metricas_grupos.csv");
    if (!grupos) throw std::runtime_error("Nao foi possivel salvar metricas por grupo");
    grupos << std::setprecision(12)
           << "grupo,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia\n";
    linha_grupo(grupos, "global", resumo.global, resumo.metricas_globais);
    linha_grupo(grupos, "forte", resumo.forte, resumo.metricas_fortes);
    linha_grupo(grupos, "fraca", resumo.fraca, resumo.metricas_fracas);
    linha_grupo(grupos, "sem_pluma", resumo.sem_pluma,
                calcular_metricas(resumo.sem_pluma));
}

#include "resultados.hpp"

#include "estatisticas.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sys/utsname.h>

#include <opencv2/core.hpp>

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

std::string distribuicao_linux() {
    std::ifstream arquivo("/etc/os-release");
    std::string linha;
    while (std::getline(arquivo, linha)) {
        constexpr const char* chave = "PRETTY_NAME=";
        if (linha.rfind(chave, 0) == 0) {
            std::string valor = linha.substr(12);
            if (valor.size() >= 2 && valor.front() == '"' && valor.back() == '"')
                valor = valor.substr(1, valor.size() - 2);
            return valor;
        }
    }
    return "indisponivel";
}

std::string versao_instalada(const fs::path& arquivo) {
    std::ifstream entrada(arquivo);
    std::string linha;
    constexpr const char* prefixo = "set(PACKAGE_VERSION \"";
    while (std::getline(entrada, linha)) {
        if (linha.rfind(prefixo, 0) == 0) {
            const auto inicio = std::string(prefixo).size();
            const auto fim = linha.find('"', inicio);
            if (fim != std::string::npos) return linha.substr(inicio, fim - inicio);
        }
    }
    return "indisponivel";
}

struct Ambiente {
    std::string distribuicao;
    std::string kernel;
    std::string arquitetura;
    std::string vitis_ai_library;
    std::string vart;
    std::string xir;
    std::string opencv;
};

const Ambiente& ambiente() {
    static const Ambiente dados = [] {
        struct utsname sistema {};
        const bool tem_uname = uname(&sistema) == 0;
        return Ambiente{
            distribuicao_linux(),
            tem_uname ? sistema.release : "indisponivel",
            tem_uname ? sistema.machine : "indisponivel",
            versao_instalada("/usr/share/cmake/vitis_ai_library/vitis_ai_library-config-version.cmake"),
            versao_instalada("/usr/share/cmake/vart/vart-config-version.cmake"),
            versao_instalada("/usr/share/cmake/xir/xir-config-version.cmake"),
            cv::getVersionString()
        };
    }();
    return dados;
}

void escrever_ambiente(std::ostream& saida) {
    const auto& a = ambiente();
    saida << ',' << csv(a.distribuicao) << ',' << csv(a.kernel)
          << ',' << csv(a.arquitetura) << ',' << csv(a.vitis_ai_library)
          << ',' << csv(a.vart) << ',' << csv(a.xir) << ',' << csv(a.opencv);
}

constexpr const char* COLUNAS_AMBIENTE =
    "linux_distribuicao,linux_kernel,arquitetura,vitis_ai_library_versao,"
    "vart_versao,xir_versao,opencv_versao";

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
                       const std::string& run_id, const fs::path& csv_dataset,
                       const ResultadoExecucao& execucao,
                       const std::vector<MedidaPotencia>& potencia,
                       bool potencia_solicitada) {
    fs::create_directories(pasta);
    const auto& c = execucao.configuracao;
    const auto r = resumir_execucao(execucao);

    auto geral = abrir(pasta / "benchmark_geral.csv",
        std::string("modelo,run_id,csv_dataset,modo,runners,nucleos_cpu,workers_pre,workers_pos,slots_por_runner,") +
        "fixar_afinidade,warmup,inferencias,entradas_preparadas,duracao_s,throughput_fps,fps_latencia,"
        "latencia_media_ms,latencia_mediana_ms,latencia_min_ms,latencia_max_ms,"
        "latencia_p95_ms,latencia_p99_ms,latencia_desvio_ms,inferencia_media_ms,"
        "leitura_media_ms,preprocess_media_ms,postprocess_media_ms," + COLUNAS_AMBIENTE);
    geral << csv(modelo) << ',' << csv(run_id) << ','
          << csv(fs::absolute(csv_dataset).string()) << ',' << execucao.modo << ','
          << c.runners << ',' << c.nucleos_cpu << ',' << c.workers_pre << ','
          << c.workers_pos << ',' << c.slots_por_runner << ','
          << c.fixar_afinidade << ',' << c.warmup << ',' << execucao.concluidas << ','
          << (execucao.modo == "model_only" ? c.runners * c.slots_por_runner : 0) << ','
          << execucao.duracao_s << ',' << execucao.throughput_fps << ','
          << (r.latencia_total.media > 0 ? 1000.0 / r.latencia_total.media : 0.0) << ','
          << r.latencia_total.media << ',' << r.latencia_total.mediana << ','
          << r.latencia_total.minimo << ',' << r.latencia_total.maximo << ','
          << r.latencia_total.p95 << ',' << r.latencia_total.p99 << ','
          << r.latencia_total.desvio_padrao << ',' << r.inferencia.media << ','
          << r.leitura.media << ',' << r.preprocess.media << ','
          << r.postprocess.media;
    escrever_ambiente(geral);
    geral << '\n';

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
        "modelo,run_id,modo,trilho,sensor_chip,fonte_name,fonte_label,fonte_power_input,"
        "status,amostras,media_w,minima_w,maxima_w,energia_j,energia_por_inferencia_j,duracao_s");
    if (potencia.empty())
        energia << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo
                << (potencia_solicitada ? ",,,,,,indisponivel,0,,,,,," :
                                         ",,,,,,desativada,0,,,,,,")
                << execucao.duracao_s << '\n';
    for (const auto& p : potencia)
        energia << csv(modelo) << ',' << csv(run_id) << ',' << execucao.modo
                << ',' << csv(p.trilho) << ',' << csv(p.sensor_chip)
                << ',' << csv(p.fonte_name) << ',' << csv(p.fonte_label)
                << ',' << csv(p.fonte_power_input) << ",ok," << p.amostras
                << ',' << p.media_w
                << ',' << p.minima_w << ',' << p.maxima_w << ',' << p.energia_j
                << ',' << (execucao.concluidas > 0 ? p.energia_j / execucao.concluidas : 0.0)
                << ',' << execucao.duracao_s << '\n';
}

void salvar_metricas(const fs::path& pasta, const std::string& modelo,
                     const std::string& run_id, const fs::path& csv_dataset,
                     const std::string& modo,
                     const ConfiguracaoPipeline& configuracao,
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
           << "modelo,run_id,csv_dataset,modo,imagens,runners,nucleos_cpu,workers_pre,workers_pos,"
           << "slots_por_runner,fixar_afinidade,warmup,inferencias_configuradas,"
           << "tp,fp,fn,tn,precision,recall,f1_global,iou,fpr,acuracia,"
           << "f1_strong_plume,f1_weak_plume,auprc,fpr_sem_pluma,"
           << "fpr_tile,fpr_tile_tabela,fp_tiles,tn_tiles,"
           << COLUNAS_AMBIENTE << '\n';
    const auto& c = resumo.global;
    const auto& m = resumo.metricas_globais;
    global << csv(modelo) << ',' << csv(run_id) << ','
           << csv(fs::absolute(csv_dataset).string()) << ',' << csv(modo)
           << ',' << resumo.imagens
           << ',' << configuracao.runners << ',' << configuracao.nucleos_cpu
           << ',' << configuracao.workers_pre << ',' << configuracao.workers_pos
           << ',' << configuracao.slots_por_runner << ','
           << configuracao.fixar_afinidade << ',' << configuracao.warmup
           << ',' << configuracao.inferencias << ',' << c.tp << ',' << c.fp
           << ',' << c.fn << ',' << c.tn << ',' << m.precision << ',' << m.recall
           << ',' << m.f1 << ',' << m.iou << ',' << m.fpr << ',' << m.acuracia
           << ',' << resumo.metricas_fortes.f1 << ',' << resumo.metricas_fracas.f1
           << ',' << resumo.auprc << ',' << resumo.fpr_sem_pluma
           << ',' << resumo.fpr_tile << ',' << resumo.fpr_tile_tabela
           << ',' << resumo.fp_tiles << ',' << resumo.tn_tiles;
    escrever_ambiente(global);
    global << '\n';

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

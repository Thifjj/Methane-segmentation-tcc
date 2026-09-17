#include "dataset.hpp"
#include "estatisticas.hpp"
#include "pipeline.hpp"
#include "power.hpp"
#include "progresso.hpp"
#include "resultados.hpp"
#include "validacao.hpp"
#include "xmodel_runner.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Opcoes {
    fs::path modelo;
    fs::path dataset;
    fs::path csv;
    fs::path saida;
    std::string run_id;
    std::string modo = "all";
    ConfiguracaoPipeline pipeline;
    std::size_t amostras = 0;
    bool potencia = true;
    int intervalo_potencia_ms = 200;
    bool validar = true;
};

void ajuda() {
    std::cout <<
        "Uso: benchmark_vitis --model ARQUIVO.xmodel --dataset PASTA --out PASTA [opcoes]\n"
        "  --csv CSV                   Padrao: DATASET/test.csv\n"
        "  --run-id ID                 Identificador nos resultados\n"
        "  --mode model_only|end_to_end|all (padrao: all)\n"
        "  --samples N                0 = todas as amostras (padrao)\n"
        "  --iterations N             0 = uma inferencia por amostra (padrao)\n"
        "  --runners 1..4             --cpu-cores 1..4 (alias: --threads)\n"
        "  --pre-workers 1..4         --post-workers 1..4\n"
        "  --slots 1..4               --warmup N\n"
        "  --pin | --no-pin           Afinidade de CPU (padrao: --no-pin)\n"
        "  --power | --no-power       Amostragem de potencia (padrao: --power)\n"
        "  --power-sample-ms N        Intervalo em ms (padrao: 200)\n"
        "  --validate | --no-validate Metricas de segmentacao (padrao: --validate)\n";
}

std::size_t numero(const std::string& texto, const std::string& opcao) {
    if (texto.empty() || texto[0] == '-')
        throw std::runtime_error("Valor invalido para " + opcao + ": " + texto);
    std::size_t fim = 0;
    unsigned long long n = 0;
    try {
        n = std::stoull(texto, &fim);
    } catch (const std::exception&) {
        throw std::runtime_error("Valor invalido para " + opcao + ": " + texto);
    }
    if (fim != texto.size() || n > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Valor invalido para " + opcao + ": " + texto);
    return static_cast<std::size_t>(n);
}

int inteiro(const std::string& texto, const std::string& opcao, int minimo, int maximo) {
    const auto n = numero(texto, opcao);
    if (n < static_cast<std::size_t>(minimo) ||
        n > static_cast<std::size_t>(maximo))
        throw std::runtime_error("Fora do intervalo em " + opcao + ": " + texto);
    return static_cast<int>(n);
}

Opcoes interpretar(int argc, char** argv) {
    Opcoes o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto valor = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Falta valor para " + a);
            return argv[i];
        };
        if (a == "--help" || a == "-h") {
            ajuda();
            std::exit(0);
        } else if (a == "--model") o.modelo = valor();
        else if (a == "--dataset") o.dataset = valor();
        else if (a == "--csv") o.csv = valor();
        else if (a == "--out") o.saida = valor();
        else if (a == "--run-id") o.run_id = valor();
        else if (a == "--mode") o.modo = valor();
        else if (a == "--samples") o.amostras = numero(valor(), a);
        else if (a == "--iterations") o.pipeline.inferencias = numero(valor(), a);
        else if (a == "--runners") o.pipeline.runners = inteiro(valor(), a, 1, 4);
        else if (a == "--cpu-cores" || a == "--threads")
            o.pipeline.nucleos_cpu = inteiro(valor(), a, 1, 4);
        else if (a == "--pre-workers")
            o.pipeline.workers_pre = inteiro(valor(), a, 1, 4);
        else if (a == "--post-workers")
            o.pipeline.workers_pos = inteiro(valor(), a, 1, 4);
        else if (a == "--slots")
            o.pipeline.slots_por_runner = inteiro(valor(), a, 1, 4);
        else if (a == "--warmup")
            o.pipeline.warmup = inteiro(valor(), a, 0, std::numeric_limits<int>::max());
        else if (a == "--pin") o.pipeline.fixar_afinidade = true;
        else if (a == "--no-pin") o.pipeline.fixar_afinidade = false;
        else if (a == "--power") o.potencia = true;
        else if (a == "--no-power") o.potencia = false;
        else if (a == "--power-sample-ms")
            o.intervalo_potencia_ms = inteiro(valor(), a, 1, std::numeric_limits<int>::max());
        else if (a == "--validate") o.validar = true;
        else if (a == "--no-validate") o.validar = false;
        else throw std::runtime_error("Opcao desconhecida: " + a);
    }

    if (o.modelo.empty() || !fs::is_regular_file(o.modelo))
        throw std::runtime_error("XModel nao encontrado: " + o.modelo.string());
    if (o.dataset.empty() || !fs::is_directory(o.dataset))
        throw std::runtime_error("Dataset nao encontrado: " + o.dataset.string());
    if (o.csv.empty()) o.csv = o.dataset / "test.csv";
    if (!fs::is_regular_file(o.csv))
        throw std::runtime_error("CSV nao encontrado: " + o.csv.string());
    if (o.saida.empty())
        throw std::runtime_error("Informe --out");
    if (o.modo != "model_only" && o.modo != "end_to_end" && o.modo != "all")
        throw std::runtime_error("--mode deve ser model_only, end_to_end ou all");
    if (o.run_id.empty())
        o.run_id = "manual_" + std::to_string(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    return o;
}

void salvar_configuracao(const Opcoes& o, std::size_t amostras,
                         const XModelRunner& modelo) {
    std::ofstream arquivo(o.saida / "config.txt");
    if (!arquivo) throw std::runtime_error("Nao foi possivel salvar config.txt");
    arquivo << "modelo=" << fs::canonical(o.modelo) << '\n'
            << "dataset=" << fs::canonical(o.dataset) << '\n'
            << "csv=" << fs::canonical(o.csv) << '\n'
            << "run_id=" << o.run_id << '\n'
            << "modo=" << o.modo << '\n'
            << "amostras=" << amostras << '\n'
            << "inferencias=" << o.pipeline.inferencias << '\n'
            << "runners=" << o.pipeline.runners << '\n'
            << "nucleos_cpu=" << o.pipeline.nucleos_cpu << '\n'
            << "workers_pre=" << o.pipeline.workers_pre << '\n'
            << "workers_pos=" << o.pipeline.workers_pos << '\n'
            << "slots_por_runner=" << o.pipeline.slots_por_runner << '\n'
            << "warmup=" << o.pipeline.warmup << '\n'
            << "fixar_afinidade=" << o.pipeline.fixar_afinidade << '\n'
            << "potencia=" << o.potencia << '\n'
            << "intervalo_potencia_ms=" << o.intervalo_potencia_ms << '\n'
            << "validar=" << o.validar << '\n'
            << "subgrafos_dpu=" << modelo.subgrafos_dpu << '\n'
            << "subgrafos_cpu=" << modelo.subgrafos_cpu << '\n'
            << "escala_entrada=" << modelo.escala_entrada << '\n'
            << "escala_saida=" << modelo.escala_saida << '\n'
            << "saida_float=" << modelo.saida_float << '\n';
    if (!arquivo) throw std::runtime_error("Falha ao gravar config.txt");
}

void executar_modo(const Opcoes& o, const std::vector<Amostra>& amostras,
                   const std::string& modo) {
    std::unique_ptr<MonitorPotencia> monitor;
    if (o.potencia)
        monitor = std::make_unique<MonitorPotencia>(o.intervalo_potencia_ms);

    std::cout << "Executando " << modo << "..." << std::endl;
    const auto execucao = [&] {
        const auto total = o.pipeline.inferencias == 0
            ? amostras.size() : o.pipeline.inferencias;
        Progresso progresso(modo, total);
        return modo == "model_only"
            ? executar_model_only(o.modelo.string(), amostras, o.pipeline,
                                  monitor.get(), progresso.contador())
            : executar_end_to_end(o.modelo.string(), amostras, o.pipeline,
                                  monitor.get(), progresso.contador());
    }();
    const auto medidas = monitor
        ? monitor->resumo(execucao.duracao_s) : std::vector<MedidaPotencia>{};
    salvar_desempenho(o.saida, o.modelo.filename().string(), o.run_id,
                       execucao, medidas, o.potencia);

    const auto estatisticas = resumir_execucao(execucao);
    std::cout << modo << ": " << execucao.concluidas << " inferencias em "
              << execucao.duracao_s << " s; " << execucao.throughput_fps
              << " FPS; latencia media " << estatisticas.latencia_total.media
              << " ms; p99 " << estatisticas.latencia_total.p99 << " ms\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = interpretar(argc, argv);
        const auto amostras = carregar_amostras(o.csv, o.dataset, o.amostras);
        if (amostras.empty()) throw std::runtime_error("CSV sem amostras");
        fs::create_directories(o.saida);

        // Inspeciona o grafo antes das regioes medidas, sem manter um runner extra.
        {
            const auto modelo = carregar_xmodel(o.modelo.string());
            salvar_configuracao(o, amostras.size(), modelo);
            std::cout << "XModel: " << modelo.subgrafos_dpu << " subgrafos DPU, "
                      << modelo.subgrafos_cpu << " CPU; " << amostras.size()
                      << " amostras\n";
        }

        if (o.modo == "model_only" || o.modo == "all")
            executar_modo(o, amostras, "model_only");
        if (o.modo == "end_to_end" || o.modo == "all")
            executar_modo(o, amostras, "end_to_end");

        if (o.validar) {
            std::cout << "Validando segmentacao..." << std::endl;
            const auto validacao = [&] {
                Progresso progresso("validacao", amostras.size());
                return validar_modelo(o.modelo.string(), amostras,
                                      o.pipeline.warmup, progresso.contador());
            }();
            salvar_metricas(o.saida, o.modelo.filename().string(),
                            validacao.imagens, validacao.resumo);
            std::cout << "F1=" << validacao.resumo.metricas_globais.f1
                      << " IoU=" << validacao.resumo.metricas_globais.iou
                      << " AUPRC=" << validacao.resumo.auprc << '\n';
        }
        std::cout << "Resultados: " << fs::absolute(o.saida) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark_vitis: " << e.what() << '\n';
        return 1;
    }
}

#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Edite somente estes valores para mudar a busca. Cada candidato executa
// o mesmo numero de inferencias; o vencedor percorre o dataset inteiro.
constexpr int PRIMEIRO_RUNNER = 2;
constexpr int ULTIMO_RUNNER = 4;
constexpr int INFERENCIAS_BUSCA = 80;
constexpr int WARMUP_BUSCA = 10;
constexpr int WARMUP_FINAL = 10;
constexpr int TIMEOUT_BUSCA_S = 90;
constexpr int TIMEOUT_FINAL_S = 3600;
constexpr const char* RAIZ_RESULTADOS = "resultados_zcu104";

struct Config {
    int runners;
    int pre_workers;
};

struct Medida {
    Config config;
    double fps = 0;
    double p99 = 0;
};

std::vector<std::string> separar_csv(const std::string& linha) {
    std::vector<std::string> campos(1);
    bool aspas = false;
    for (std::size_t i = 0; i < linha.size(); ++i) {
        const char c = linha[i];
        if (c == '"') {
            if (aspas && i + 1 < linha.size() && linha[i + 1] == '"') {
                campos.back() += '"';
                ++i;
            } else aspas = !aspas;
        } else if (c == ',' && !aspas) campos.emplace_back();
        else campos.back() += c;
    }
    if (aspas) throw std::runtime_error("CSV de resultado com aspas abertas");
    return campos;
}

std::pair<double, double> ler_resultado(const fs::path& arquivo,
                                       const std::string& run_id,
                                       const std::string& modo) {
    std::ifstream entrada(arquivo);
    std::string linha;
    if (!entrada || !std::getline(entrada, linha))
        throw std::runtime_error("Resultado ausente: " + arquivo.string());
    const auto cabecalho = separar_csv(linha);
    const auto coluna = [&](const std::string& nome) {
        const auto it = std::find(cabecalho.begin(), cabecalho.end(), nome);
        if (it == cabecalho.end()) throw std::runtime_error("Coluna ausente: " + nome);
        return static_cast<std::size_t>(it - cabecalho.begin());
    };
    const auto id = coluna("run_id");
    const auto modo_col = coluna("modo");
    const auto fps = coluna("throughput_fps");
    const auto p99 = coluna("latencia_p99_ms");
    while (std::getline(entrada, linha)) {
        const auto campos = separar_csv(linha);
        if (campos.size() == cabecalho.size() && campos[id] == run_id &&
            campos[modo_col] == modo)
            return {std::stod(campos[fps]), std::stod(campos[p99])};
    }
    throw std::runtime_error("Execucao nao encontrada: " + run_id);
}

int executar_processo(const std::vector<std::string>& argumentos, int timeout_s) {
    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("fork falhou");
    if (pid == 0) {
        std::vector<char*> ptrs;
        for (const auto& arg : argumentos) ptrs.push_back(const_cast<char*>(arg.c_str()));
        ptrs.push_back(nullptr);
        execv(ptrs[0], ptrs.data());
        _exit(127);
    }
    const auto limite = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_s);
    int status = 0;
    while (true) {
        const pid_t fim = waitpid(pid, &status, WNOHANG);
        if (fim == pid) break;
        if (fim < 0) throw std::runtime_error("waitpid falhou");
        if (std::chrono::steady_clock::now() >= limite) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            throw std::runtime_error("Benchmark excedeu o tempo limite");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

Medida medir(const fs::path& benchmark, const fs::path& modelo,
             const fs::path& dataset, const fs::path& saida,
             const std::string& modo, Config config, bool final) {
    const std::string run_id = (final ? "final_" : "busca_") + modo +
        "_r" + std::to_string(config.runners) +
        "_pre" + std::to_string(config.pre_workers);
    const fs::path pasta = saida / (final ? "final" : "runs") / run_id;
    std::vector<std::string> args = {
        benchmark.string(), "--model", modelo.string(),
        "--dataset", dataset.string(),
        "--internal-out", pasta.string(),
        "--internal-run-id", run_id,
        "--internal-mode", modo,
        "--internal-runners", std::to_string(config.runners),
        "--internal-pre-workers", std::to_string(config.pre_workers),
        "--internal-iterations", std::to_string(
            final ? 0 : INFERENCIAS_BUSCA),
        "--internal-warmup", std::to_string(final ? WARMUP_FINAL : WARMUP_BUSCA)
    };
    if (!final) {
        args.emplace_back("--internal-no-power");
        args.emplace_back("--internal-no-validate");
    }
    std::cout << run_id << "..." << std::endl;
    const int status = executar_processo(args, final ? TIMEOUT_FINAL_S : TIMEOUT_BUSCA_S);
    if (status != 0)
        throw std::runtime_error(run_id + " falhou com status " + std::to_string(status));
    const auto [fps, p99] = ler_resultado(
        pasta / "benchmark_geral.csv", run_id,
        modo == "all" ? "end_to_end" : modo);
    if (fps <= 0 || p99 < 0) throw std::runtime_error("Medida invalida: " + run_id);
    std::cout << run_id << ": " << fps << " FPS, P99 " << p99 << " ms\n";
    return {config, fps, p99};
}

bool melhor(const Medida& a, const Medida& b) {
    return a.fps > b.fps || (a.fps == b.fps && a.p99 < b.p99);
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path modelo, dataset;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") {
                std::cout << "Uso: sweep_vitis --model ARQUIVO.xmodel --dataset PASTA\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + a);
            if (a == "--model") modelo = argv[++i];
            else if (a == "--dataset") dataset = argv[++i];
            else throw std::runtime_error("Opcao desconhecida: " + a);
        }
        if (!fs::is_regular_file(modelo) || !fs::is_directory(dataset) ||
            (!fs::is_regular_file(dataset / "test.csv") &&
             !fs::is_regular_file(dataset / "train.csv")))
            throw std::runtime_error("Modelo, dataset ou CSV nao encontrado");
        const fs::path benchmark = fs::absolute(argv[0]).parent_path() / "benchmark_vitis";
        if (!fs::is_regular_file(benchmark))
            throw std::runtime_error("Compile benchmark_vitis antes do sweep");

        const auto instante = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const fs::path saida = fs::path(RAIZ_RESULTADOS) /
            (modelo.stem().string() + "_" + dataset.filename().string() +
             "_sweep_" + std::to_string(instante));
        fs::create_directories(saida);
        std::ofstream ranking(saida / "ranking_search.csv");
        if (!ranking) throw std::runtime_error("Nao foi possivel gravar ranking");
        ranking << "modo,runners,nucleos_cpu,workers_pre,workers_pos,slots,throughput_fps,p99_ms\n";

        // Tres runners travaram a placa com MobileNetV2 em execucoes anteriores.
        std::string nome_modelo = modelo.stem().string();
        std::transform(nome_modelo.begin(), nome_modelo.end(), nome_modelo.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        const int ultimo = nome_modelo.find("mobilenet_v2") != std::string::npos
            ? 2 : ULTIMO_RUNNER;
        Medida melhor_e2e{};
        bool achou_e2e = false;
        const auto tentar = [&](const std::string& modo, Config c,
                                Medida& vencedor, bool& achou) {
            try {
                const auto m = medir(benchmark, modelo, dataset, saida, modo, c, false);
                ranking << modo << ',' << c.runners << ",4," << c.pre_workers
                        << ",1,2," << m.fps << ',' << m.p99 << '\n';
                ranking.flush();
                if (!achou || melhor(m, vencedor)) vencedor = m;
                achou = true;
            } catch (const std::exception& e) {
                std::cerr << "Candidato ignorado: " << e.what() << '\n';
            }
        };
        for (int runners = PRIMEIRO_RUNNER; runners <= ultimo; ++runners) {
            for (int pre : {2, 4})
                tentar("end_to_end", {runners, pre}, melhor_e2e, achou_e2e);
        }
        if (!achou_e2e)
            throw std::runtime_error("Nenhuma configuracao valida");

        std::ofstream best(saida / "best_config.txt");
        best << "CPU=4, post_workers=1, slots=2\n"
             << "end_to_end: runners=" << melhor_e2e.config.runners
             << " pre_workers=" << melhor_e2e.config.pre_workers << '\n';
        medir(benchmark, modelo, dataset, saida,
              "all", melhor_e2e.config, true);
        std::cout << "Resultados: " << fs::absolute(saida) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sweep_vitis: " << e.what() << '\n';
        return 1;
    }
}

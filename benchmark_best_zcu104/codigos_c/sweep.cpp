#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

struct Opcoes {
    fs::path modelo;
    fs::path dataset;
    fs::path csv;
    fs::path saida;
    fs::path benchmark;
    int max_runners = 4;
    int search_iterations = 50;
    int candidate_iterations = 150;
    int confirm_repeats = 3;
    int final_repeats = 3;
    int warmup = 5;
    int final_warmup = 20;
    int timeout_seconds = 3600;
    bool resume = false;
};

struct Config {
    int runners = 1;
    int cores = 4;
    int pre = 1;
    int pos = 1;
    int slots = 2;
    bool pin = false;
};

struct Medida {
    Config config;
    std::string fase;
    std::string modo;
    double fps = 0;
    double p99 = 0;
};

std::vector<std::string> separar_csv(const std::string& linha) {
    std::vector<std::string> campos(1);
    bool aspas = false;
    for (std::size_t i = 0; i < linha.size(); ++i) {
        if (linha[i] == '"') {
            if (aspas && i + 1 < linha.size() && linha[i + 1] == '"') {
                campos.back() += '"';
                ++i;
            } else aspas = !aspas;
        } else if (linha[i] == ',' && !aspas) campos.emplace_back();
        else campos.back() += linha[i];
    }
    if (aspas) throw std::runtime_error("CSV de resultado com aspas abertas");
    return campos;
}

std::string id(const std::string& fase, const std::string& modo,
               const Config& c, int repeticao) {
    return fase + "__" + modo + "__r" + std::to_string(c.runners) +
           "_cpu" + std::to_string(c.cores) + "_pre" + std::to_string(c.pre) +
           "_pos" + std::to_string(c.pos) + "_s" + std::to_string(c.slots) +
           (c.pin ? "_pin" : "_nopin") + "_rep" + std::to_string(repeticao);
}

Opcoes interpretar(int argc, char** argv) {
    Opcoes o;
    o.benchmark = fs::absolute(argv[0]).parent_path() / "benchmark_vitis";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto valor = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Falta valor para " + a);
            return argv[i];
        };
        if (a == "--model") o.modelo = valor();
        else if (a == "--dataset") o.dataset = valor();
        else if (a == "--csv") o.csv = valor();
        else if (a == "--out") o.saida = valor();
        else if (a == "--benchmark") o.benchmark = valor();
        else if (a == "--max-runners") o.max_runners = std::stoi(valor());
        else if (a == "--search-iterations") o.search_iterations = std::stoi(valor());
        else if (a == "--candidate-iterations") o.candidate_iterations = std::stoi(valor());
        else if (a == "--confirm-repeats") o.confirm_repeats = std::stoi(valor());
        else if (a == "--final-repeats") o.final_repeats = std::stoi(valor());
        else if (a == "--warmup") o.warmup = std::stoi(valor());
        else if (a == "--final-warmup") o.final_warmup = std::stoi(valor());
        else if (a == "--timeout-seconds") o.timeout_seconds = std::stoi(valor());
        else if (a == "--resume") o.resume = true;
        else if (a == "--help") {
            std::cout << "Uso: sweep_vitis --model ARQUIVO.xmodel --dataset PASTA "
                         "--out PASTA [--csv CSV] [--max-runners 1..4] "
                         "[--search-iterations N] [--candidate-iterations N] "
                         "[--confirm-repeats N] [--final-repeats N] [--resume]\n";
            std::exit(0);
        } else throw std::runtime_error("Opcao desconhecida: " + a);
    }
    if (o.csv.empty()) o.csv = o.dataset / "test.csv";
    if (o.modelo.empty() || o.dataset.empty() || o.saida.empty() ||
        !fs::is_regular_file(o.modelo) || !fs::is_directory(o.dataset) ||
        !fs::is_regular_file(o.csv) || !fs::is_regular_file(o.benchmark) ||
        o.max_runners < 1 || o.max_runners > 4 || o.search_iterations < 1 ||
        o.candidate_iterations < 1 || o.confirm_repeats < 1 ||
        o.final_repeats < 1 || o.warmup < 0 || o.final_warmup < 0 ||
        o.timeout_seconds < 1)
        throw std::runtime_error("Caminhos ou limites do sweep invalidos");
    return o;
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

std::pair<double, double> ler_resultado(const fs::path& arquivo,
                                         const std::string& run_id,
                                         const std::string& modo) {
    std::ifstream entrada(arquivo);
    std::string linha;
    if (!entrada || !std::getline(entrada, linha))
        throw std::runtime_error("Resumo ausente: " + arquivo.string());
    const auto cabecalho = separar_csv(linha);
    const auto indice = [&](const std::string& nome) {
        const auto it = std::find(cabecalho.begin(), cabecalho.end(), nome);
        if (it == cabecalho.end()) throw std::runtime_error("Coluna ausente: " + nome);
        return static_cast<std::size_t>(it - cabecalho.begin());
    };
    const auto id_col = indice("run_id");
    const auto modo_col = indice("modo");
    const auto fps_col = indice("throughput_fps");
    const auto p99_col = indice("latencia_p99_ms");
    while (std::getline(entrada, linha)) {
        const auto campos = separar_csv(linha);
        if (campos.size() == cabecalho.size() && campos[id_col] == run_id &&
            campos[modo_col] == modo) {
            const double fps = std::stod(campos[fps_col]);
            const double p99 = std::stod(campos[p99_col]);
            if (fps > 0 && p99 >= 0) return {fps, p99};
        }
    }
    throw std::runtime_error("Resultado invalido: " + arquivo.string());
}

Medida medir(const Opcoes& o, const std::string& fase, const std::string& modo,
             Config c, int repeticao, int iteracoes, int warmup,
             bool potencia, bool validar, std::vector<Medida>& historico) {
    const std::string run_id = id(fase, modo, c, repeticao);
    const fs::path pasta = o.saida / (fase == "final" ? "final" : "runs") / run_id;
    const fs::path resumo = pasta / "benchmark_geral.csv";
    std::pair<double, double> valores;
    bool reutilizado = false;
    if (o.resume && fs::exists(resumo) &&
        (!validar || fs::exists(pasta / "metricas_globais.csv")) &&
        (!potencia || fs::exists(pasta / "benchmark_power_rails.csv"))) {
        try {
            valores = ler_resultado(resumo, run_id, modo);
            reutilizado = true;
        } catch (const std::exception&) {
            // Uma execucao interrompida fica preservada e e repetida.
        }
    }
    if (!reutilizado) {
        if (fs::exists(pasta)) {
            if (!o.resume)
                throw std::runtime_error("Pasta ja existe: " + pasta.string() +
                                         "; use --resume ou outro --out");
            int tentativa = 1;
            fs::path preservada;
            do {
                preservada = pasta.string() + ".incompleta_" +
                             std::to_string(tentativa++);
            } while (fs::exists(preservada));
            fs::rename(pasta, preservada);
        }
        fs::create_directories(pasta);
        std::vector<std::string> args = {
            fs::absolute(o.benchmark).string(), "--model", fs::absolute(o.modelo).string(),
            "--dataset", fs::absolute(o.dataset).string(), "--csv", fs::absolute(o.csv).string(),
            "--out", fs::absolute(pasta).string(), "--run-id", run_id,
            "--mode", modo, "--runners", std::to_string(c.runners),
            "--cpu-cores", std::to_string(c.cores), "--pre-workers", std::to_string(c.pre),
            "--post-workers", std::to_string(c.pos), "--slots", std::to_string(c.slots),
            "--iterations", std::to_string(iteracoes), "--warmup", std::to_string(warmup),
            c.pin ? "--pin" : "--no-pin", potencia ? "--power" : "--no-power",
            validar ? "--validate" : "--no-validate"
        };
        std::cout << "[" << run_id << "] iniciando" << std::endl;
        const int status = executar_processo(args, o.timeout_seconds);
        if (status != 0)
            throw std::runtime_error(run_id + " falhou com status " + std::to_string(status));
        valores = ler_resultado(resumo, run_id, modo);
    }
    Medida m{c, fase, modo, valores.first, valores.second};
    historico.push_back(m);
    std::cout << "[" << run_id << "] FPS=" << m.fps << " P99=" << m.p99
              << (reutilizado ? " (resume)" : "") << std::endl;
    return m;
}

bool melhor(const Medida& a, const Medida& b) {
    if (a.fps != b.fps) return a.fps > b.fps;
    return a.p99 < b.p99;
}

std::vector<Medida> melhores(std::vector<Medida> medidas, std::size_t quantidade) {
    std::sort(medidas.begin(), medidas.end(), melhor);
    if (medidas.size() > quantidade) medidas.resize(quantidade);
    return medidas;
}

Medida confirmar(const Opcoes& o, const std::string& modo,
                 const std::vector<Medida>& candidatos,
                 std::vector<Medida>& historico) {
    std::vector<Medida> confirmados;
    for (const auto& candidato : candidatos) {
        for (bool pin : {false, true}) {
            Config c = candidato.config;
            c.pin = pin;
            std::vector<double> fps;
            std::vector<double> p99;
            for (int rep = 1; rep <= o.confirm_repeats; ++rep) {
                auto m = medir(o, "confirm", modo, c, rep, o.candidate_iterations,
                               o.warmup, false, false, historico);
                fps.push_back(m.fps);
                p99.push_back(m.p99);
            }
            std::sort(fps.begin(), fps.end());
            std::sort(p99.begin(), p99.end());
            confirmados.push_back({c, "confirm", modo, fps[fps.size() / 2],
                                   p99[p99.size() / 2]});
        }
    }
    return melhores(std::move(confirmados), 1).front();
}

void salvar_ranking(const Opcoes& o, const std::vector<Medida>& historico,
                    const Medida& modelo, const Medida& e2e) {
    std::ofstream ranking(o.saida / "ranking_search.csv");
    ranking << "fase,modo,runners,nucleos_cpu,workers_pre,workers_pos,slots,pin,fps,p99_ms\n"
            << std::setprecision(12);
    for (const auto& m : historico)
        ranking << m.fase << ',' << m.modo << ',' << m.config.runners << ','
                << m.config.cores << ',' << m.config.pre << ',' << m.config.pos << ','
                << m.config.slots << ',' << m.config.pin << ',' << m.fps << ','
                << m.p99 << '\n';

    std::ofstream best(o.saida / "best_config.txt");
    best << std::setprecision(12);
    for (const auto& m : {modelo, e2e})
        best << m.modo << ": runners=" << m.config.runners
             << " cpu=" << m.config.cores << " pre=" << m.config.pre
             << " post=" << m.config.pos << " slots=" << m.config.slots
             << " pin=" << m.config.pin << " fps_confirmado=" << m.fps
             << " p99_ms=" << m.p99 << '\n';
}

void verificar_campanha(const Opcoes& o) {
    const fs::path arquivo = o.saida / "campanha.txt";
    const std::string esperado =
        "modelo=" + fs::canonical(o.modelo).string() + "\n" +
        "dataset=" + fs::canonical(o.dataset).string() + "\n" +
        "csv=" + fs::canonical(o.csv).string() + "\n" +
        "max_runners=" + std::to_string(o.max_runners) + "\n" +
        "search_iterations=" + std::to_string(o.search_iterations) + "\n" +
        "candidate_iterations=" + std::to_string(o.candidate_iterations) + "\n" +
        "confirm_repeats=" + std::to_string(o.confirm_repeats) + "\n" +
        "final_repeats=" + std::to_string(o.final_repeats) + "\n" +
        "warmup=" + std::to_string(o.warmup) + "\n" +
        "final_warmup=" + std::to_string(o.final_warmup) + "\n";
    if (fs::exists(arquivo)) {
        std::ifstream entrada(arquivo);
        const std::string atual((std::istreambuf_iterator<char>(entrada)),
                                std::istreambuf_iterator<char>());
        if (atual != esperado)
            throw std::runtime_error("Campanha existente tem parametros diferentes: " +
                                     arquivo.string());
    } else {
        std::ofstream saida(arquivo);
        if (!saida || !(saida << esperado))
            throw std::runtime_error("Nao foi possivel escrever " + arquivo.string());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = interpretar(argc, argv);
        fs::create_directories(o.saida);
        verificar_campanha(o);
        std::vector<Medida> historico;
        std::vector<Medida> grade_modelo;
        std::vector<Medida> grade_e2e;

        for (int cpu = 1; cpu <= 4; ++cpu) {
            for (int runners = 1; runners <= o.max_runners; ++runners) {
                Config c{runners, cpu, 1, 1, 1, false};
                grade_modelo.push_back(medir(o, "model_grid", "model_only", c, 1,
                    o.search_iterations, o.warmup, false, false, historico));
                c.slots = 2;
                grade_e2e.push_back(medir(o, "e2e_grid", "end_to_end", c, 1,
                    o.search_iterations, o.warmup, false, false, historico));
            }
        }

        const auto melhor_modelo = confirmar(o, "model_only",
            melhores(grade_modelo, 3), historico);

        std::vector<Medida> ajustados;
        for (const auto& base : melhores(grade_e2e, 3)) {
            for (int pre = 1; pre <= 4; ++pre) {
                for (int pos = 1; pos <= 4; ++pos) {
                    Config c = base.config;
                    c.pre = pre;
                    c.pos = pos;
                    ajustados.push_back(medir(o, "e2e_workers", "end_to_end", c,
                        1, o.search_iterations, o.warmup, false, false, historico));
                }
            }
        }
        std::vector<Medida> slots_ajustados;
        for (const auto& base : melhores(ajustados, 3)) {
            for (int slots = 1; slots <= 4; ++slots) {
                Config c = base.config;
                c.slots = slots;
                slots_ajustados.push_back(medir(o, "e2e_slots", "end_to_end", c,
                    1, o.search_iterations, o.warmup, false, false, historico));
            }
        }
        const auto melhor_e2e = confirmar(o, "end_to_end",
            melhores(slots_ajustados, 3), historico);

        salvar_ranking(o, historico, melhor_modelo, melhor_e2e);
        for (int rep = 1; rep <= o.final_repeats; ++rep) {
            medir(o, "final", "model_only", melhor_modelo.config, rep, 0,
                  o.final_warmup, true, false, historico);
            medir(o, "final", "end_to_end", melhor_e2e.config, rep, 0,
                  o.final_warmup, true, rep == 1, historico);
        }
        std::cout << "Sweep concluido. Configuracoes em "
                  << (o.saida / "best_config.txt") << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
        return 1;
    }
}

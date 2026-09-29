#include <core/session/onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include "power.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/utsname.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int LADO = 512;
constexpr size_t PIXELS = LADO * LADO;

struct Opcoes {
    fs::path modelo, dataset, csv, saida;
    size_t limite = 0;
    int warmup = 10, threads = 4, intervalo_potencia_ms = 200;
    std::string modo = "all";
    bool potencia = true;
};

struct Amostra {
    fs::path pasta;
    bool has_plume;
    double qplume;
    cv::Rect janela;
};

struct Contagens {
    uint64_t tp = 0, fp = 0, fn = 0, tn = 0;
    void somar(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        tp += a; fp += b; fn += c; tn += d;
    }
};

std::vector<std::string> separar_csv(const std::string& linha) {
    std::vector<std::string> campos(1);
    bool aspas = false;
    for (size_t i = 0; i < linha.size(); ++i) {
        char c = linha[i];
        if (c == '"') {
            if (aspas && i + 1 < linha.size() && linha[i + 1] == '"') {
                campos.back() += '"'; ++i;
            } else aspas = !aspas;
        } else if (c == ',' && !aspas) campos.emplace_back();
        else campos.back() += c;
    }
    if (aspas) throw std::runtime_error("Aspas nao fechadas no CSV");
    return campos;
}

std::string csv_val(const std::string& valor) {
    std::string texto = "\"";
    for (char c : valor) {
        if (c == '"') texto += '"';
        texto += c;
    }
    return texto + '"';
}

std::string ambiente_csv() {
    std::ifstream arquivo("/etc/os-release");
    std::string linha, distribuicao = "indisponivel";
    while (std::getline(arquivo, linha)) {
        if (linha.rfind("PRETTY_NAME=", 0) == 0) {
            distribuicao = linha.substr(12);
            if (distribuicao.size() >= 2 && distribuicao.front() == '"' && distribuicao.back() == '"')
                distribuicao = distribuicao.substr(1, distribuicao.size() - 2);
            break;
        }
    }
    struct utsname sistema {};
    const bool ok = uname(&sistema) == 0;
    return csv_val(distribuicao) + ',' + csv_val(ok ? sistema.release : "indisponivel") + ','
           + csv_val(ok ? sistema.machine : "indisponivel") + ','
           + csv_val(OrtGetApiBase()->GetVersionString()) + ',' + csv_val(cv::getVersionString());
}

Opcoes argumentos(int argc, char** argv) {
    Opcoes o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Uso: benchmark_arm --model ARQUIVO.onnx --dataset PASTA [--csv ARQUIVO] [--output PASTA] [--limit N] [--warmup N] [--threads N] [--mode all|model_only|end_to_end] [--power-interval-ms N] [--no-power]\n";
            std::exit(0);
        }
        if (a == "--no-power") { o.potencia = false; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Falta valor para " + a);
        std::string v = argv[++i];
        if (a == "--model") o.modelo = v;
        else if (a == "--dataset") o.dataset = v;
        else if (a == "--csv") o.csv = v;
        else if (a == "--output") o.saida = v;
        else if (a == "--limit") o.limite = std::stoul(v);
        else if (a == "--warmup") o.warmup = std::stoi(v);
        else if (a == "--threads") o.threads = std::stoi(v);
        else if (a == "--mode") o.modo = v;
        else if (a == "--power-interval-ms") o.intervalo_potencia_ms = std::stoi(v);
        else throw std::runtime_error("Opcao desconhecida: " + a);
    }
    if (!fs::is_regular_file(o.modelo)) throw std::runtime_error("Modelo nao encontrado");
    if (!fs::is_directory(o.dataset)) throw std::runtime_error("Dataset nao encontrado");
    if (!o.csv.empty() && o.csv.is_relative()) o.csv = o.dataset / o.csv;
    if (o.csv.empty()) {
        bool teste = fs::is_regular_file(o.dataset / "test.csv");
        bool treino = fs::is_regular_file(o.dataset / "train.csv");
        if (teste == treino) throw std::runtime_error("Indique --csv test.csv ou train.csv");
        o.csv = o.dataset / (teste ? "test.csv" : "train.csv");
    }
    if (!fs::is_regular_file(o.csv)) throw std::runtime_error("CSV nao encontrado");
    if (o.warmup < 0 || o.threads < 1 || o.intervalo_potencia_ms < 1)
        throw std::runtime_error("warmup/threads/intervalo invalidos");
    if (o.modo != "all" && o.modo != "model_only" && o.modo != "end_to_end")
        throw std::runtime_error("Modo invalido");
    if (o.saida.empty()) {
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        o.saida = fs::path("resultados_arm") /
                  (o.modelo.stem().string() + "_" + o.dataset.filename().string() + "_" + std::to_string(stamp));
    }
    return o;
}

std::vector<Amostra> amostras(const Opcoes& o) {
    std::ifstream arquivo(o.csv);
    std::string linha;
    std::getline(arquivo, linha);
    auto colunas = separar_csv(linha);
    if (!colunas.empty() && colunas.front().rfind("\xEF\xBB\xBF", 0) == 0)
        colunas.front().erase(0, 3);
    auto it = std::find(colunas.begin(), colunas.end(), "folder");
    if (it == colunas.end()) throw std::runtime_error("CSV sem coluna folder");
    const size_t indice = it - colunas.begin();
    auto indice_coluna = [&](const char* nome) {
        auto pos = std::find(colunas.begin(), colunas.end(), nome);
        if (pos == colunas.end()) throw std::runtime_error(std::string("CSV sem coluna ") + nome);
        return static_cast<size_t>(pos - colunas.begin());
    };
    const size_t indice_pluma = indice_coluna("has_plume");
    const size_t indice_qplume = indice_coluna("qplume");
    const bool tem_janela = std::find(colunas.begin(), colunas.end(), "window_col_off") != colunas.end();
    size_t ix = 0, iy = 0, iw = 0, ih = 0;
    if (tem_janela) {
        ix = indice_coluna("window_col_off");
        iy = indice_coluna("window_row_off");
        iw = indice_coluna("window_width");
        ih = indice_coluna("window_height");
    }
    std::vector<Amostra> pastas;
    while (std::getline(arquivo, linha)) {
        auto campos = separar_csv(linha);
        if (std::max({indice, indice_pluma, indice_qplume, tem_janela ? ix : 0,
                      tem_janela ? iy : 0, tem_janela ? iw : 0, tem_janela ? ih : 0}) >= campos.size())
            throw std::runtime_error("Linha CSV incompleta");
        fs::path pasta = o.dataset / fs::path(campos[indice]).filename();
        const auto& p = campos[indice_pluma];
        if (p != "True" && p != "False" && p != "true" && p != "false" && p != "1" && p != "0")
            throw std::runtime_error("has_plume invalido");
        bool has_plume = p == "True" || p == "true" || p == "1";
        if (has_plume && campos[indice_qplume].empty())
            throw std::runtime_error("qplume ausente para amostra com pluma");
        double qplume = campos[indice_qplume].empty() ? 0.0 : std::stod(campos[indice_qplume]);
        if (has_plume && !std::isfinite(qplume)) throw std::runtime_error("qplume invalido");
        cv::Rect janela;
        if (tem_janela) janela = cv::Rect(std::stoi(campos[ix]), std::stoi(campos[iy]),
                                         std::stoi(campos[iw]), std::stoi(campos[ih]));
        if (tem_janela && (janela.width <= 0 || janela.height <= 0))
            throw std::runtime_error("Janela invalida");
        if (fs::is_directory(pasta)) pastas.push_back({pasta, has_plume, qplume, janela});
        if (o.limite && pastas.size() >= o.limite) break;
    }
    if (pastas.empty()) throw std::runtime_error("Nenhuma amostra encontrada");
    return pastas;
}

cv::Mat ler(const Amostra& amostra, const char* nome) {
    cv::Mat m = cv::imread((amostra.pasta / nome).string(), cv::IMREAD_UNCHANGED);
    if (m.empty() || m.channels() != 1)
        throw std::runtime_error("TIFF invalido: " + (amostra.pasta / nome).string());
    if (amostra.janela.area() > 0) {
        if (amostra.janela.x < 0 || amostra.janela.y < 0 ||
            amostra.janela.x + amostra.janela.width > m.cols ||
            amostra.janela.y + amostra.janela.height > m.rows)
            throw std::runtime_error("Janela fora do TIFF");
        m = m(amostra.janela);
    }
    if (m.rows != LADO || m.cols != LADO) throw std::runtime_error("Esperado tile 512x512");
    cv::Mat f;
    m.convertTo(f, CV_32F);
    return f;
}

using Canais = std::array<cv::Mat, 4>;

Canais carregar_canais(const Amostra& amostra) {
    constexpr const char* nomes[] = {"mag1c.tif", "TOA_AVIRIS_640nm.tif", "TOA_AVIRIS_550nm.tif", "TOA_AVIRIS_460nm.tif"};
    return {ler(amostra, nomes[0]), ler(amostra, nomes[1]),
            ler(amostra, nomes[2]), ler(amostra, nomes[3])};
}

void preprocessar(const Canais& canais, std::vector<float>& entrada) {
    for (int c = 0; c < 4; ++c) {
        const cv::Mat& canal = canais[c];
        const float divisor = c == 0 ? 1750.0f : 60.0f;
        for (size_t p = 0; p < PIXELS; ++p) {
            const float valor = canal.ptr<float>()[p];
            if (!std::isfinite(valor)) throw std::runtime_error("Canal nao finito");
            entrada[c * PIXELS + p] = std::clamp(valor / divisor, 0.0f, 2.0f);
        }
    }
}

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void mostrar_progresso(const char* etapa, size_t feitas, size_t total) {
    std::cout << '\r' << etapa << ": " << feitas << '/' << total
              << " (" << 100 * feitas / total << "%)" << std::flush;
    if (feitas == total) std::cout << '\n';
}

struct Estatisticas {
    double media = 0, mediana = 0, minimo = 0, maximo = 0, p95 = 0, p99 = 0, desvio = 0;
};

Estatisticas resumir(std::vector<double> valores) {
    std::sort(valores.begin(), valores.end());
    if (valores.empty()) return {};
    const auto percentil = [&](double p) {
        double pos = p * (valores.size() - 1);
        size_t i = static_cast<size_t>(pos);
        return valores[i] * (1 - (pos - i)) + valores[std::min(i + 1, valores.size() - 1)] * (pos - i);
    };
    double media = std::accumulate(valores.begin(), valores.end(), 0.0) / valores.size();
    double quadrados = 0;
    for (double v : valores) quadrados += (v - media) * (v - media);
    return {media, percentil(0.5), valores.front(), valores.back(), percentil(0.95),
            percentil(0.99), std::sqrt(quadrados / valores.size())};
}

double razao(double numerador, double denominador) {
    return denominador ? numerador / denominador : 0.0;
}

double f1(const Contagens& c) {
    return razao(2.0 * c.tp, 2.0 * c.tp + c.fp + c.fn);
}

void salvar_potencia(const Opcoes& o, const std::string& modo, const MonitorPotencia* monitor,
                     double duracao_s, size_t inferencias) {
    std::ofstream arquivo(o.saida / "benchmark_power_rails.csv", std::ios::app);
    if (!arquivo) throw std::runtime_error("Falha ao salvar potencia");
    if (arquivo.tellp() == 0)
        arquivo << "modelo,modo,trilho,sensor_chip,fonte_name,fonte_label,fonte_power_input,status,amostras,media_w,minima_w,maxima_w,energia_j,energia_por_inferencia_j,duracao_s\n";
    auto medidas = monitor ? monitor->resumo(duracao_s) : std::vector<MedidaPotencia>{};
    if (medidas.empty()) {
        arquivo << o.modelo.filename().string() << ',' << modo << ",,,,,,"
                << (o.potencia ? "indisponivel" : "desativada") << ",0,,,,,," << duracao_s << '\n';
    }
    for (const auto& p : medidas) {
        arquivo << csv_val(o.modelo.filename().string()) << ',' << modo << ',' << csv_val(p.trilho) << ','
                << csv_val(p.sensor_chip) << ',' << csv_val(p.fonte_name) << ',' << csv_val(p.fonte_label) << ','
                << csv_val(p.fonte_power_input) << ",ok," << p.amostras << ',' << p.media_w << ','
                << p.minima_w << ',' << p.maxima_w << ',' << p.energia_j << ','
                << razao(p.energia_j, inferencias) << ',' << duracao_s << '\n';
    }
}

void salvar_desempenho(const Opcoes& o, const std::string& modo,
                       const std::vector<double>& leitura, const std::vector<double>& pre,
                       const std::vector<double>& inferencia, const std::vector<double>& pos,
                       const std::vector<double>& total, double duracao_s) {
    const auto m = resumir(inferencia), e = resumir(total);
    std::ofstream geral(o.saida / "benchmark_geral.csv", std::ios::app);
    if (!geral) throw std::runtime_error("Falha ao salvar desempenho");
    if (geral.tellp() == 0)
        geral << "modelo,csv_dataset,modo,threads,warmup,ort_optimizacao,cpu_mem_arena,mem_pattern,inferencias,duracao_s,throughput_fps,fps_latencia,latencia_media_ms,latencia_mediana_ms,latencia_min_ms,latencia_max_ms,latencia_p95_ms,latencia_p99_ms,latencia_desvio_ms,inferencia_media_ms,leitura_media_ms,preprocess_media_ms,postprocess_media_ms,linux_distribuicao,linux_kernel,arquitetura,onnxruntime_versao,opencv_versao\n";
    geral << std::setprecision(12) << o.modelo.filename().string() << ',' << o.csv.string() << ','
          << modo << ',' << o.threads << ',' << o.warmup << ",basic,0,0," << inferencia.size() << ','
          << duracao_s << ',' << razao(inferencia.size(), duracao_s) << ','
          << razao(1000.0, e.media) << ',' << e.media << ',' << e.mediana << ',' << e.minimo
          << ',' << e.maximo << ',' << e.p95 << ',' << e.p99 << ',' << e.desvio << ','
          << m.media << ',' << resumir(leitura).media << ',' << resumir(pre).media << ','
          << resumir(pos).media << ',' << ambiente_csv() << '\n';

    std::ofstream estagios(o.saida / "benchmark_estagios.csv", std::ios::app);
    if (!estagios) throw std::runtime_error("Falha ao salvar estagios");
    if (estagios.tellp() == 0)
        estagios << "modelo,modo,estagio,amostras,media_ms,mediana_ms,minimo_ms,maximo_ms,p95_ms,p99_ms,desvio_padrao_ms\n";
    const auto linha = [&](const char* nome, const std::vector<double>& valores) {
        if (valores.empty()) return;
        const auto s = resumir(valores);
        estagios << o.modelo.filename().string() << ',' << modo << ',' << nome << ','
                 << valores.size() << ',' << s.media << ',' << s.mediana << ',' << s.minimo
                 << ',' << s.maximo << ',' << s.p95 << ',' << s.p99 << ',' << s.desvio << '\n';
    };
    linha("latencia_total", total);
    linha("leitura", leitura);
    linha("preprocess", pre);
    linha("inferencia", inferencia);
    linha("postprocess", pos);
}

int main(int argc, char** argv) {
    const char* fase = "argumentos";
    try {
        const auto o = argumentos(argc, argv);
        fase = "leitura do CSV";
        const auto pastas = amostras(o);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark_arm");
        Ort::SessionOptions opcoes;
        opcoes.SetIntraOpNumThreads(o.threads);
        opcoes.SetInterOpNumThreads(1);
        opcoes.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        opcoes.DisableCpuMemArena();
        opcoes.DisableMemPattern();
        fase = "carregamento do ONNX";
        Ort::Session sessao(env, o.modelo.c_str(), opcoes);
        // O exportador fixa nomes, tipo float32 e shapes dos dois tensores.
        const char* entradas[] = {"input"};
        const char* saidas[] = {"logits"};
        fase = "alocacao dos buffers";
        std::vector<float> dados(4 * PIXELS);
        std::vector<float> logits(PIXELS);
        const int64_t shape_entrada[] = {1, 4, LADO, LADO};
        const int64_t shape_saida[] = {1, 1, LADO, LADO};
        fase = "criacao do MemoryInfo";
        auto memoria = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        fase = "criacao dos tensores";
        auto tensor_entrada = Ort::Value::CreateTensor<float>(
            memoria, dados.data(), dados.size(), shape_entrada, 4);
        auto tensor_saida = Ort::Value::CreateTensor<float>(
            memoria, logits.data(), logits.size(), shape_saida, 4);
        auto inferir = [&] {
            sessao.Run(Ort::RunOptions{nullptr}, entradas, &tensor_entrada, 1,
                       saidas, &tensor_saida, 1);
        };
        fase = "preprocessamento inicial";
        preprocessar(carregar_canais(pastas.front()), dados);
        fase = "warmup";
        if (o.warmup) mostrar_progresso("warmup", 0, o.warmup);
        for (int i = 0; i < o.warmup; ++i) {
            inferir();
            mostrar_progresso("warmup", i + 1, o.warmup);
        }
        fs::create_directories(o.saida);
        for (const char* nome : {"benchmark_geral.csv", "benchmark_estagios.csv", "benchmark_power_rails.csv"})
            std::ofstream(o.saida / nome, std::ios::trunc);
        std::ofstream amostras_csv(o.saida / "benchmark_samples.csv");
        if (!amostras_csv) throw std::runtime_error("Falha ao salvar amostras");
        amostras_csv << "modelo,modo,id,leitura_ms,preprocess_ms,inferencia_ms,postprocess_ms,latencia_total_ms,pixels_previstos\n";

        const auto executar = [&](const std::string& modo) {
            std::vector<double> leitura, pre, inferencia, pos, total;
            leitura.reserve(pastas.size()); pre.reserve(pastas.size());
            inferencia.reserve(pastas.size()); pos.reserve(pastas.size()); total.reserve(pastas.size());
            std::unique_ptr<MonitorPotencia> monitor;
            if (o.potencia) monitor = std::make_unique<MonitorPotencia>(o.intervalo_potencia_ms);
            if (monitor) monitor->iniciar();
            const auto inicio = Clock::now();
            size_t concluidas = 0;
            mostrar_progresso(modo.c_str(), 0, pastas.size());
            for (const auto& amostra : pastas) {
                const auto t0 = Clock::now();
                if (modo == "end_to_end") {
                    auto canais = carregar_canais(amostra);
                    const auto t1 = Clock::now();
                    preprocessar(canais, dados);
                    const auto t2 = Clock::now();
                    inferir();
                    const auto t3 = Clock::now();
                    size_t previstos = 0;
                    for (size_t p = 0; p < PIXELS; ++p) {
                        if (!std::isfinite(logits[p])) throw std::runtime_error("Logit nao finito");
                        previstos += logits[p] > 0.0f;
                    }
                    const auto t4 = Clock::now();
                    leitura.push_back(ms(t0, t1)); pre.push_back(ms(t1, t2));
                    inferencia.push_back(ms(t2, t3)); pos.push_back(ms(t3, t4)); total.push_back(ms(t0, t4));
                    amostras_csv << o.modelo.filename().string() << ',' << modo << ',' << amostra.pasta.filename().string()
                                 << ',' << leitura.back() << ',' << pre.back() << ',' << inferencia.back()
                                 << ',' << pos.back() << ',' << total.back() << ',' << previstos << '\n';
                } else {
                    inferir();
                    const auto t1 = Clock::now();
                    inferencia.push_back(ms(t0, t1)); total.push_back(inferencia.back());
                    amostras_csv << o.modelo.filename().string() << ',' << modo << ',' << pastas.front().pasta.filename().string()
                                 << ",0,0," << inferencia.back() << ",0," << total.back() << ",0\n";
                }
                mostrar_progresso(modo.c_str(), ++concluidas, pastas.size());
            }
            const auto fim = Clock::now();
            if (monitor) monitor->parar();
            double duracao_s = ms(inicio, fim) / 1000.0;
            salvar_desempenho(o, modo, leitura, pre, inferencia, pos, total, duracao_s);
            salvar_potencia(o, modo, monitor.get(), duracao_s, pastas.size());
            std::cout << modo << ": " << resumir(total).media << " ms; "
                      << razao(pastas.size(), duracao_s) << " FPS\n";
        };

        if (o.modo == "all" || o.modo == "model_only") {
            fase = "model_only";
            executar("model_only");
        }
        if (o.modo == "all" || o.modo == "end_to_end") {
            fase = "end_to_end";
            executar("end_to_end");
        }

        // Validacao separada: leitura de labels e metricas nao contaminam energia/latencia.
        fase = "validacao";
        Contagens global, forte, fraca, sem_pluma;
        uint64_t fp_tiles = 0, tn_tiles = 0;
        std::array<uint64_t, 256> positivos{}, negativos{};
        std::ofstream por_imagem(o.saida / "metricas_por_imagem.csv");
        if (!por_imagem) throw std::runtime_error("Falha ao salvar metricas");
        por_imagem << "modelo,id,dificuldade,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia\n";
        size_t validadas = 0;
        mostrar_progresso("validacao", 0, pastas.size());
        for (const auto& amostra : pastas) {
            preprocessar(carregar_canais(amostra), dados);
            inferir();
            cv::Mat label = ler(amostra, "labelbinary.tif");
            Contagens c;
            uint64_t previstos = 0;
            for (size_t p = 0; p < PIXELS; ++p) {
                if (!std::isfinite(logits[p])) throw std::runtime_error("Logit nao finito");
                if (!std::isfinite(label.ptr<float>()[p])) throw std::runtime_error("Label nao finito");
                bool pred = logits[p] > 0.0f;
                bool real = label.ptr<float>()[p] != 0.0f;
                previstos += pred;
                c.somar(pred && real, pred && !real, !pred && real, !pred && !real);
                const int bin = static_cast<int>(std::lrint(std::clamp(logits[p] / 0.1f, -128.0f, 127.0f))) + 128;
                (real ? positivos : negativos)[bin]++;
            }
            global.somar(c.tp, c.fp, c.fn, c.tn);
            std::string grupo;
            if (!amostra.has_plume) {
                grupo = "sem_pluma";
                sem_pluma.somar(c.tp, c.fp, c.fn, c.tn);
                if (previstos > 10 * PIXELS / (64 * 64)) ++fp_tiles; else ++tn_tiles;
            } else if (amostra.qplume > 1000.0) {
                grupo = "forte";
                forte.somar(c.tp, c.fp, c.fn, c.tn);
            } else {
                grupo = "fraca";
                fraca.somar(c.tp, c.fp, c.fn, c.tn);
            }
            por_imagem << o.modelo.filename().string() << ',' << amostra.pasta.filename().string() << ',' << grupo
                       << ',' << c.tp << ',' << c.fp << ',' << c.fn << ',' << c.tn
                       << ',' << razao(c.tp, c.tp + c.fp) << ',' << razao(c.tp, c.tp + c.fn)
                       << ',' << f1(c) << ',' << razao(c.tp, c.tp + c.fp + c.fn)
                       << ',' << razao(c.fp, c.fp + c.tn)
                       << ',' << razao(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn) << '\n';
            mostrar_progresso("validacao", ++validadas, pastas.size());
        }

        const uint64_t total_positivos = std::accumulate(positivos.begin(), positivos.end(), uint64_t{0});
        uint64_t tp_pr = total_positivos;
        uint64_t fp_pr = std::accumulate(negativos.begin(), negativos.end(), uint64_t{0});
        double auprc = total_positivos ? 0.0 : 0.5;
        if (total_positivos) for (size_t bin = 0; bin < positivos.size(); ++bin) {
            double antes = razao(tp_pr, tp_pr + fp_pr);
            tp_pr -= positivos[bin]; fp_pr -= negativos[bin];
            double depois = tp_pr + fp_pr ? razao(tp_pr, tp_pr + fp_pr) : 1.0;
            auprc += razao(positivos[bin], total_positivos) * (antes + depois) / 2.0;
        }
        std::ofstream metricas(o.saida / "metricas_globais.csv");
        if (!metricas) throw std::runtime_error("Falha ao salvar metricas globais");
        metricas << "modelo,csv_dataset,modo,imagens,threads,warmup,tp,fp,fn,tn,precision,recall,f1_global,iou,fpr,acuracia,f1_strong_plume,f1_weak_plume,auprc,fpr_sem_pluma,fpr_tile,fpr_tile_tabela,fp_tiles,tn_tiles,linux_distribuicao,linux_kernel,arquitetura,onnxruntime_versao,opencv_versao\n";
        metricas << std::setprecision(12) << o.modelo.filename().string() << ',' << o.csv.string()
                 << ',' << o.modo << ',' << pastas.size() << ',' << o.threads << ',' << o.warmup
                 << ',' << global.tp << ',' << global.fp << ',' << global.fn << ',' << global.tn
                 << ',' << razao(global.tp, global.tp + global.fp)
                 << ',' << razao(global.tp, global.tp + global.fn) << ',' << f1(global)
                 << ',' << razao(global.tp, global.tp + global.fp + global.fn)
                 << ',' << razao(global.fp, global.fp + global.tn)
                 << ',' << razao(global.tp + global.tn, global.tp + global.fp + global.fn + global.tn)
                 << ',' << f1(forte) << ',' << f1(fraca) << ',' << auprc
                 << ',' << razao(sem_pluma.fp, sem_pluma.fp + sem_pluma.tn)
                 << ',' << razao(fp_tiles, fp_tiles + tn_tiles)
                 << ',' << razao(fp_tiles, pastas.size()) << ',' << fp_tiles << ',' << tn_tiles
                 << ',' << ambiente_csv() << '\n';
        std::ofstream grupos(o.saida / "metricas_grupos.csv");
        if (!grupos) throw std::runtime_error("Falha ao salvar metricas por grupo");
        grupos << "grupo,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia\n";
        const auto linha_grupo = [&](const char* nome, const Contagens& c) {
            grupos << nome << ',' << c.tp << ',' << c.fp << ',' << c.fn << ',' << c.tn
                   << ',' << razao(c.tp, c.tp + c.fp) << ',' << razao(c.tp, c.tp + c.fn)
                   << ',' << f1(c) << ',' << razao(c.tp, c.tp + c.fp + c.fn)
                   << ',' << razao(c.fp, c.fp + c.tn)
                   << ',' << razao(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn) << '\n';
        };
        linha_grupo("global", global); linha_grupo("forte", forte);
        linha_grupo("fraca", fraca); linha_grupo("sem_pluma", sem_pluma);
        std::cout << "F1=" << f1(global) << " AUPRC=" << auprc
                  << " FPR tile=" << razao(fp_tiles, fp_tiles + tn_tiles)
                  << "\nResultados: " << o.saida << '\n';
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "benchmark_arm: falha de alocacao em %s (std::bad_alloc)\n", fase);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "benchmark_arm: " << e.what() << '\n';
        return 1;
    }
}

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#else
#include <core/session/onnxruntime_cxx_api.h>
#endif
#include "dataset.hpp"
#include "postprocess.hpp"
#include "metricas.hpp"
#include "power.hpp"
#include "sha256.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/utsname.h>
#include <sched.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr std::size_t PIXELS = PIXELS_SAIDA;
struct Opcoes
{
    fs::path modelo, dataset, csv, saida;
    std::size_t limite = 0, inferencias = 0;
    int warmup = 10, threads = 4, intervalo_potencia_ms = 200, patch_size = 0, patch_batch_size = 1;
    std::string modo = "all", ordem = "auto", run_id;
    bool potencia = true;
};
std::string csv_val(const std::string &valor)
{
    std::string texto = "\"";
    for (char c : valor)
    {
        if (c == '"')
            texto += '"';
        texto += c;
    }
    return texto + '"';
}

std::string ambiente_csv()
{
    std::ifstream arquivo("/etc/os-release");
    std::string linha, distribuicao = "indisponivel";
    while (std::getline(arquivo, linha))
    {
        if (linha.rfind("PRETTY_NAME=", 0) == 0)
        {
            distribuicao = linha.substr(12);
            if (distribuicao.size() >= 2 && distribuicao.front() == '"' && distribuicao.back() == '"')
                distribuicao = distribuicao.substr(1, distribuicao.size() - 2);
            break;
        }
    }
    struct utsname sistema{};
    const bool ok = uname(&sistema) == 0;
    return csv_val(distribuicao) + ',' + csv_val(ok ? sistema.release : "indisponivel") + ',' + csv_val(ok ? sistema.machine : "indisponivel") + ',' + csv_val(OrtGetApiBase()->GetVersionString()) + ',' + csv_val(cv::getVersionString());
}

double ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void mostrar_progresso(const char *etapa, size_t feitas, size_t total)
{
    std::cout << '\r' << etapa << ": " << feitas << '/' << total
              << " (" << 100 * feitas / total << "%)" << std::flush;
    if (feitas == total)
        std::cout << '\n';
}

struct Estatisticas
{
    double media = 0, mediana = 0, minimo = 0, maximo = 0, p95 = 0, p99 = 0, desvio = 0;
};

Estatisticas resumir(std::vector<double> valores)
{
    std::sort(valores.begin(), valores.end());
    if (valores.empty())
        return {};
    const auto percentil = [&](double p)
    {
        double pos = p * (valores.size() - 1);
        size_t i = static_cast<size_t>(pos);
        return valores[i] * (1 - (pos - i)) + valores[std::min(i + 1, valores.size() - 1)] * (pos - i);
    };
    double media = std::accumulate(valores.begin(), valores.end(), 0.0) / valores.size();
    double quadrados = 0;
    for (double v : valores)
        quadrados += (v - media) * (v - media);
    return {media, percentil(0.5), valores.front(), valores.back(), percentil(0.95),
            percentil(0.99), std::sqrt(quadrados / valores.size())};
}

std::size_t inteiro(const std::string &v)
{
    if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Esperado inteiro nao negativo: " + v);
    return std::stoull(v);
}
Opcoes argumentos(int argc, char **argv)
{
    Opcoes o;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h")
        {
            std::cout << "Uso: benchmark_arm --model ARQUIVO.onnx --dataset PASTA [--csv ARQUIVO] [--output PASTA]\n"
                         "  --limit N --inferencias N --warmup N --threads N --mode all|model_only|end_to_end\n"
                         "  --patch-size 128|512 (auto pelo ONNX) --patch-batch-size N --channel-order auto|rgb|legacy\n"
                         "  --power-interval-ms N --no-power\n";
            std::exit(0);
        }
        if (a == "--no-power")
        {
            o.potencia = false;
            continue;
        }
        if (i + 1 >= argc)
            throw std::runtime_error("Falta valor para " + a);
        const std::string v = argv[++i];
        if (a == "--model")
            o.modelo = v;
        else if (a == "--dataset")
            o.dataset = v;
        else if (a == "--csv")
            o.csv = v;
        else if (a == "--output")
            o.saida = v;
        else if (a == "--mode")
            o.modo = v;
        else if (a == "--channel-order")
            o.ordem = v;
        else if (a == "--limit")
            o.limite = inteiro(v);
        else if (a == "--inferencias")
            o.inferencias = inteiro(v);
        else
        {
            auto n = inteiro(v);
            if (n > 1000000)
                throw std::runtime_error("Opcao numerica acima de 1000000");
            if (a == "--warmup")
                o.warmup = n;
            else if (a == "--threads")
                o.threads = n;
            else if (a == "--patch-size")
                o.patch_size = n;
            else if (a == "--patch-batch-size")
                o.patch_batch_size = n;
            else if (a == "--power-interval-ms")
                o.intervalo_potencia_ms = n;
            else
                throw std::runtime_error("Opcao desconhecida: " + a);
        }
    }
    if (!fs::is_regular_file(o.modelo) || !fs::is_directory(o.dataset))
        throw std::runtime_error("Modelo ou dataset nao encontrado");
    if (o.csv.empty())
    {
        bool test = fs::is_regular_file(o.dataset / "test.csv"), train = fs::is_regular_file(o.dataset / "train.csv");
        if (test == train)
            throw std::runtime_error("Indique --csv test.csv ou train.csv");
        o.csv = o.dataset / (test ? "test.csv" : "train.csv");
    }
    else if (o.csv.is_relative())
        o.csv = o.dataset / o.csv;
    if (!fs::is_regular_file(o.csv))
        throw std::runtime_error("CSV nao encontrado");
    if (o.threads < 1 || o.intervalo_potencia_ms < 1 || o.patch_batch_size < 1 ||
        (o.patch_size != 0 && o.patch_size != 128 && o.patch_size != 512))
        throw std::runtime_error("Configuracao numerica invalida");
    if (o.modo != "all" && o.modo != "model_only" && o.modo != "end_to_end")
        throw std::runtime_error("Modo invalido");
    if (o.ordem != "auto" && o.ordem != "rgb" && o.ordem != "legacy")
        throw std::runtime_error("Ordem de canais invalida");
    o.run_id = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    if (o.saida.empty())
        o.saida = fs::path("resultados_arm") / (o.modelo.stem().string() + "_" + o.dataset.filename().string() + "_" + o.run_id);
    if (fs::exists(o.saida) && (!fs::is_directory(o.saida) || !fs::is_empty(o.saida)))
        throw std::runtime_error("Pasta de resultados ja contem arquivos; escolha outra --output");
    return o;
}

struct ModeloONNX
{
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "benchmark_arm"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string input_name, output_name, checkpoint_hash, parametros, checkpoint, modelo;
    std::vector<float> patches_saida;
    int patch = 512, patches = 1, batch = 1;
    bool rgb = false;
    explicit ModeloONNX(const Opcoes &o)
    {
        options.SetIntraOpNumThreads(o.threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        options.DisableCpuMemArena();
        options.DisableMemPattern();
        session = std::make_unique<Ort::Session>(env, o.modelo.c_str(), options);
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
            throw std::runtime_error("ONNX deve ter uma entrada e uma saida");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session->GetInputNameAllocated(0, allocator).get();
        output_name = session->GetOutputNameAllocated(0, allocator).get();
        auto it = session->GetInputTypeInfo(0), ot = session->GetOutputTypeInfo(0);
        const auto info = it.GetTensorTypeAndShapeInfo(), out = ot.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape(), output = out.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape.size() != 4 || output.size() != 4 || shape[1] != 4 || output[1] != 1 || shape[2] != shape[3] ||
            (shape[2] != 128 && shape[2] != 512) || output[2] != shape[2] || output[3] != shape[3])
            throw std::runtime_error("Contrato ONNX: FP32 NCHW [B,4,P,P] -> [B,1,P,P], P=128 ou 512");
        patch = shape[2];
        patches = (512 / patch) * (512 / patch);
        batch = o.patch_batch_size;
        if (o.patch_size && o.patch_size != patch)
            throw std::runtime_error("--patch-size difere do modelo ONNX");
        if (batch > patches || (shape[0] > 0 && (shape[0] != batch || patches % batch)) ||
            (output[0] > 0 && (output[0] != batch || patches % batch)))
            throw std::runtime_error("Batch incompativel; exporte --dynamic-batch para agrupar patches");
        if (shape[0] == 0 || output[0] == 0)
            throw std::runtime_error("Batch ONNX invalido");
        auto metadata = session->GetModelMetadata();
        const auto prop = [&](const char *key)
        {auto s=metadata.LookupCustomMetadataMapAllocated(key,allocator);return s?std::string(s.get()):std::string(); };
        auto channels = prop("ordem_canais");
        if (!channels.empty() && channels != "mag1c,460,550,640" && channels != "mag1c,640,550,460")
            throw std::runtime_error("Ordem de canais ONNX desconhecida");
        rgb = o.ordem == "rgb" || (o.ordem == "auto" && channels == "mag1c,460,550,640");
        checkpoint_hash = prop("checkpoint_sha256");
        parametros = prop("parametros");
        checkpoint = prop("checkpoint");
        modelo = prop("modelo");
        if (modelo.empty())
            modelo = o.modelo.stem().string();
        patches_saida.resize(PIXELS);
    }
    void inferir(const std::vector<float> &inputs)
    {
        if (inputs.size() != 4 * PIXELS)
            throw std::runtime_error("Buffer de entrada invalido");
        const char *entradas[] = {input_name.c_str()};
        const char *saidas[] = {output_name.c_str()};
        const std::size_t pp = patch * patch;
        for (int offset = 0; offset < patches; offset += batch)
        {
            const int n = std::min(batch, patches - offset);
            const std::array<int64_t, 4> si{n, 4, patch, patch}, so{n, 1, patch, patch};
            auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float *>(inputs.data()) + offset * 4 * pp, n * 4 * pp, si.data(), 4);
            auto output = Ort::Value::CreateTensor<float>(memory, patches_saida.data() + offset * pp, n * pp, so.data(), 4);
            session->Run(Ort::RunOptions{nullptr}, entradas, &input, 1, saidas, &output, 1);
        }
    }
};

void preparar(const CanaisEntrada &channels, std::vector<float> &inputs, int patch)
{
    inputs.resize(4 * PIXELS);
    const int grid = 512 / patch;
    for (int c = 0; c < 4; ++c)
        for (int y = 0; y < 512; ++y)
            for (int x = 0; x < 512; ++x)
            {
                const float v = channels[c].ptr<float>(y)[x];
                if (!std::isfinite(v))
                    throw std::runtime_error("Canal nao finito");
                const std::size_t idx = ((y / patch) * grid + x / patch) * 4 * patch * patch + c * patch * patch + (y % patch) * patch + x % patch;
                inputs[idx] = std::clamp(v / (c == 0 ? 1750.0f : 60.0f), 0.0f, 2.0f);
            }
}

std::string geometria(const ModeloONNX &m)
{
    return "512,512," + std::to_string(m.patch) + "," + std::to_string(m.patches) + "," + std::to_string(m.batch) + "," +
           csv_val(m.rgb ? "mag1c,460,550,640" : "mag1c,640,550,460") + ",imagem_512x512,onnxruntime_sequencial,FP32,cpu,0,1,1,1,0";
}
const char *geometry_header = "imagem_altura,imagem_largura,tamanho_patch,patches_por_imagem,patch_batch_size,ordem_canais,unidade,protocolo,precisao,device,runners,instancias_modelo,workers_pre,workers_pos,slots_por_runner";
std::string identidade(const Opcoes &o, const ModeloONNX &m)
{
    return csv_val(m.modelo) + ',' + csv_val(o.run_id) + ',' + csv_val(o.csv.string());
}
void escrever_valores(std::ostream &f, const Contagens &c, bool oficial = false)
{
    f << c.tp << ',' << c.fp << ',' << c.fn << ',' << c.tn;
    for (auto v : metricas(c, oficial))
        f << ',' << v;
}
void salvar_metricas(const fs::path &dir, const Opcoes &o, const ModeloONNX &m,
                     const std::vector<ResultadoImagem> &rows, const ResumoMetricas &r, bool oficial, double auc)
{
    fs::create_directories(dir);
    std::ofstream f(dir / "metricas_globais.csv"), g(dir / "metricas_grupos.csv"), im(dir / "metricas_por_imagem.csv");
    if (!f || !g || !im)
        throw std::runtime_error("Falha ao escrever metricas");
    f << "modelo,run_id,csv_dataset,imagens,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia,f1_global,f1_strong_plume,f1_weak_plume,auprc,fpr_sem_pluma,fpr_tile,fpr_tile_tabela,fp_tiles,tn_tiles,imagens_positivas,auprc_metodo,posprocessamento,grupo_forte,protocolo,limiar_pixels_tile\n";
    f << std::setprecision(12) << identidade(o, m) << ',' << r.imagens << ',';
    escrever_valores(f, r.global, oficial);
    const double ap = oficial ? (r.imagens_positivas ? r.soma_ap / r.imagens_positivas : std::numeric_limits<double>::quiet_NaN()) : auc;
    f << ',' << metricas(r.global, oficial)[2] << ',' << metricas(r.forte, oficial)[2] << ',' << metricas(r.fraca, oficial)[2] << ',' << ap
      << ',' << razao(r.sem_pluma.fp, r.sem_pluma.fp + r.sem_pluma.tn + (oficial ? 1e-6 : 0))
      << ',' << razao(r.fp_tiles, r.fp_tiles + r.tn_tiles) << ',' << razao(r.fp_tiles, r.imagens)
      << ',' << r.fp_tiles << ',' << r.tn_tiles << ',' << r.imagens_positivas << ','
      << (oficial ? "media_average_precision_imagens_positivas,abertura_cruz_3x3,label_positivo_e_difficulty_easy,vitis_ai_evaluate_quantized" : "auc_pr_global_pixels,logit_maior_zero,has_plume_e_qplume_maior_1000,benchmark_zcu104_bruto") << ",640\n";
    g << "grupo,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia\n"
      << std::setprecision(12);
    for (auto item : std::vector<std::pair<std::string, Contagens>>{{"global", r.global}, {"forte", r.forte}, {"fraca", r.fraca}, {"sem_pluma", r.sem_pluma}})
    {
        g << item.first << ',';
        escrever_valores(g, item.second, oficial);
        g << '\n';
    }
    im << "id,difficulty,positive,difficulty_csv,pixels_preditos,tp,fp,fn,tn,precision,recall,f1,iou,fpr,acuracia,average_precision\n"
       << std::setprecision(12);
    for (const auto &row : rows)
    {
        im << csv_val(row.id) << ',' << row.dificuldade << ',' << row.positive << ',' << csv_val(row.difficulty_csv) << ',' << row.pixels_preditos << ',';
        escrever_valores(im, row.c);
        im << ',' << row.average_precision << '\n';
    }
    if (!f || !g || !im)
        throw std::runtime_error("Falha na escrita das metricas");
    std::cout << (oficial ? "Oficial" : "Bruto") << ": F1=" << metricas(r.global, oficial)[2] << " AUPRC=" << ap << '\n';
}

void config(const Opcoes &o, const ModeloONNX &m, std::size_t imagens, const std::string &model_hash,
            const std::string &csv_hash, const std::string &status)
{
    cv::FileStorage f((o.saida / "config.json").string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!f.isOpened())
        throw std::runtime_error("Falha ao salvar config.json");
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    const int allowed = sched_getaffinity(0, sizeof(cpus), &cpus) == 0 ? CPU_COUNT(&cpus) : 0;
    struct utsname os{};
    uname(&os);
    f << "modelo" << m.modelo << "run_id" << o.run_id << "model_path" << fs::absolute(o.modelo).string() << "onnx_sha256" << model_hash
      << "checkpoint" << m.checkpoint << "checkpoint_sha256" << m.checkpoint_hash << "parametros" << m.parametros
      << "csv_dataset" << fs::absolute(o.csv).string() << "csv_sha256" << csv_hash << "imagens" << double(imagens)
      << "imagem_altura" << 512 << "imagem_largura" << 512 << "tamanho_patch" << m.patch << "patches_por_imagem" << m.patches
      << "patch_batch_size" << m.batch << "ordem_canais" << (m.rgb ? "mag1c,460,550,640" : "mag1c,640,550,460")
      << "unidade" << "imagem_512x512" << "protocolo" << "onnxruntime_sequencial" << "precisao" << "FP32" << "device" << "cpu"
      << "threads" << o.threads << "cpus_permitidas" << allowed << "warmup" << o.warmup << "inferencias_configuradas" << double(o.inferencias)
      << "runners" << 0 << "instancias_modelo" << 1 << "workers_pre" << 1 << "workers_pos" << 1 << "slots_por_runner" << 0
      << "onnxruntime_versao" << OrtGetApiBase()->GetVersionString() << "opencv_versao" << cv::getVersionString()
      << "linux_kernel" << os.release << "arquitetura" << os.machine
      << "ort_optimizacao" << "basic" << "cpu_mem_arena" << 0 << "mem_pattern" << 0
      << "model_only_entradas" << "primeiras_4_preparadas_reutilizadas" << "e2e_exclui" << "label,metricas,escrita_csv"
      << "potencia_solicitada" << int(o.potencia) << "power_interval_ms" << o.intervalo_potencia_ms
      << "normalizacao" << "mag1c/1750;bandas/60;clip[0,2]" << "status" << status;
}

void executar(const Opcoes &o, ModeloONNX &m, const std::vector<Amostra> &samples,
              const std::vector<std::vector<float>> &prepared, const std::string &mode)
{
    const std::size_t n = o.inferencias ? o.inferencias : samples.size();
    // Todas as alocacoes de buffers/telemetria ficam fora da janela medida.
    std::vector<float> input(4 * PIXELS), logits(PIXELS);
    std::vector<std::uint8_t> mask(PIXELS);
    std::vector<std::array<double, 5>> tempos(n);
    std::vector<std::size_t> indices(n);
    std::unique_ptr<MonitorPotencia> power;
    if (o.potencia)
        power = std::make_unique<MonitorPotencia>(o.intervalo_potencia_ms);
    if (power)
        power->iniciar();
    mostrar_progresso(mode.c_str(), 0, n);
    const auto inicio = Clock::now();
    for (std::size_t i = 0; i < n; ++i)
    {
        indices[i] = i % (mode == "model_only" ? prepared.size() : samples.size());
        const auto t0 = Clock::now();
        if (mode == "model_only")
        {
            m.inferir(prepared[indices[i]]);
            const auto t1 = Clock::now();
            tempos[i] = {0, 0, ms(t0, t1), 0, ms(t0, t1)};
        }
        else
        {
            auto channels = carregar_canais(samples[indices[i]], m.rgb);
            const auto t1 = Clock::now();
            preparar(channels, input, m.patch);
            const auto t2 = Clock::now();
            m.inferir(input);
            const auto t3 = Clock::now();
            reconstruir_patches(m.patches_saida.data(), logits.data(), m.patch, sizeof(float));
            posprocessar(logits.data(), logits.size() * sizeof(float), true, mask);
            const auto t4 = Clock::now();
            tempos[i] = {ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t0, t4)};
        }
        if (i + 1 == n || (i + 1) % 10 == 0)
            mostrar_progresso(mode.c_str(), i + 1, n);
    }
    const auto fim = Clock::now();
    if (power)
        power->parar();
    const double duration = ms(inicio, fim) / 1000;
    std::array<std::vector<double>, 5> stages;
    for (const auto &t : tempos)
        for (int j = 0; j < 5; ++j)
            stages[j].push_back(t[j]);
    const auto lat = resumir(stages[4]);
    std::ofstream geral(o.saida / "benchmark_geral.csv", std::ios::app), est(o.saida / "benchmark_estagios.csv", std::ios::app),
        ims(o.saida / "benchmark_samples.csv", std::ios::app), pwr(o.saida / "benchmark_power_rails.csv", std::ios::app);
    if (!geral || !est || !ims || !pwr)
        throw std::runtime_error("Falha ao abrir resultados");
    if (geral.tellp() == 0)
        geral << "modelo,run_id,csv_dataset,modo,threads,warmup,ort_optimizacao,cpu_mem_arena,mem_pattern,inferencias,entradas_preparadas,duracao_s,throughput_fps,fps_latencia,latencia_media_ms,latencia_mediana_ms,latencia_min_ms,latencia_max_ms,latencia_p95_ms,latencia_p99_ms,latencia_desvio_ms,inferencia_media_ms,leitura_media_ms,preprocess_media_ms,postprocess_media_ms,execucoes_modelo,sync_entrada_media_ms,sync_saida_media_ms," << geometry_header << ",linux_distribuicao,linux_kernel,arquitetura,onnxruntime_versao,opencv_versao\n";
    geral << std::setprecision(12) << identidade(o, m) << ',' << mode << ',' << o.threads << ',' << o.warmup << ",basic,0,0," << n << ','
          << (mode == "model_only" ? prepared.size() : 0) << ',' << duration << ',' << n / duration << ',' << razao(1000, lat.media)
          << ',' << lat.media << ',' << lat.mediana << ',' << lat.minimo << ',' << lat.maximo << ',' << lat.p95 << ',' << lat.p99 << ',' << lat.desvio
          << ',' << resumir(stages[2]).media << ',' << resumir(stages[0]).media << ',' << resumir(stages[1]).media << ',' << resumir(stages[3]).media
          << ',' << n * ((m.patches + m.batch - 1) / m.batch) << ",0,0," << geometria(m) << ',' << ambiente_csv() << '\n';
    if (est.tellp() == 0)
        est << "modelo,run_id,modo,estagio,amostras,media_ms,mediana_ms,minimo_ms,maximo_ms,p95_ms,p99_ms,desvio_padrao_ms\n";
    const char *names[] = {"leitura", "preprocess", "inferencia", "postprocess", "latencia_total"};
    est << std::setprecision(12);
    for (int j = 0; j < 7; ++j)
    {
        const auto s = j < 5 ? resumir(stages[j]) : Estatisticas{};
        est << csv_val(m.modelo) << ',' << csv_val(o.run_id) << ',' << mode << ',' << (j < 5 ? names[j] : j == 5 ? "sync_entrada"
                                                                                                                 : "sync_saida")
            << ',' << n
            << ',' << s.media << ',' << s.mediana << ',' << s.minimo << ',' << s.maximo << ',' << s.p95 << ',' << s.p99 << ',' << s.desvio << '\n';
    }
    if (ims.tellp() == 0)
        ims << "modelo,run_id,modo,trabalho,indice_amostra,id,leitura_ms,preprocess_ms,inferencia_ms,postprocess_ms,latencia_total_ms,sync_entrada_ms,sync_saida_ms\n";
    ims << std::setprecision(12);
    for (std::size_t i = 0; i < n; ++i)
    {
        ims << csv_val(m.modelo) << ',' << csv_val(o.run_id) << ',' << mode << ',' << i << ',' << indices[i] << ',' << csv_val(samples[indices[i]].id);
        for (auto t : tempos[i])
            ims << ',' << t;
        ims << ",0,0\n";
    }
    if (pwr.tellp() == 0)
        pwr << "modelo,run_id,modo,trilho,sensor_chip,fonte_name,fonte_label,fonte_power_input,status,amostras,media_w,minima_w,maxima_w,energia_j,energia_por_inferencia_j,duracao_s,metodo,unidade\n";
    const auto measures = power ? power->resumo(inicio, fim) : std::vector<MedidaPotencia>{};
    pwr << std::setprecision(12);
    if (measures.empty())
        pwr << csv_val(m.modelo) << ',' << csv_val(o.run_id) << ',' << mode << ",,,,,," << (o.potencia ? "indisponivel" : "desativada") << ",0,,,,,," << duration << ",integral_trapezoidal_hwmon,imagem_512x512\n";
    for (const auto &p : measures)
        pwr << csv_val(m.modelo) << ',' << csv_val(o.run_id) << ',' << mode << ',' << csv_val(p.trilho) << ',' << csv_val(p.sensor_chip)
            << ',' << csv_val(p.fonte_name) << ',' << csv_val(p.fonte_label) << ',' << csv_val(p.fonte_power_input) << ",ok," << p.amostras
            << ',' << p.media_w << ',' << p.minima_w << ',' << p.maxima_w << ',' << p.energia_j << ',' << p.energia_j / n << ',' << duration << ",integral_trapezoidal_hwmon,imagem_512x512\n";
    if (!geral || !est || !ims || !pwr)
        throw std::runtime_error("Falha na escrita dos resultados");
    std::cout << mode << ": " << lat.media << " ms/imagem; " << n / duration << " imagens/s\n";
}

int main(int argc, char **argv)
{
    const char *fase = "argumentos";
    fs::path temporarios, output;
    try
    {
        auto o = argumentos(argc, argv);
        fase = "carregamento do ONNX";
        ModeloONNX m(o);
        fase = "leitura do dataset";
        const auto samples = carregar_amostras(o.csv, o.dataset, o.limite, true);
        // Exige difficulty somente para validacao oficial; nao inventa classificacao.
        std::ifstream csv(o.csv);
        std::string header;
        std::getline(csv, header);
        if (!std::regex_search(header, std::regex("(^|,)\"?difficulty\"?(,|\\r?$)")))
            throw std::runtime_error("Validacao oficial exige coluna difficulty");
        fase = "preparacao das quatro entradas";
        std::vector<std::vector<float>> prepared(std::min(std::size_t(4), samples.size()));
        for (std::size_t i = 0; i < prepared.size(); ++i)
            preparar(carregar_canais(samples[i], m.rgb), prepared[i], m.patch);
        fase = "warmup";
        for (int i = 0; i < o.warmup; ++i)
        {
            m.inferir(prepared[0]);
            mostrar_progresso("warmup", i + 1, o.warmup);
        }
        fs::create_directories(o.saida);
        output = o.saida;
        const auto model_hash = arquivo_sha256(o.modelo), csv_hash = arquivo_sha256(o.csv);
        config(o, m, samples.size(), model_hash, csv_hash, "em_execucao");
        std::ofstream manifest(o.saida / "amostras.csv");
        manifest << "id,folder,difficulty,has_plume,qplume\n";
        for (const auto &s : samples)
            manifest << csv_val(s.id) << ',' << csv_val(s.pasta.string()) << ',' << csv_val(s.difficulty) << ',' << s.has_plume << ',' << s.qplume << '\n';
        manifest.close();
        if (!manifest)
            throw std::runtime_error("Falha ao salvar amostras");
        std::cout << m.modelo << ": " << samples.size() << " imagens; patch=" << m.patch << " batch=" << m.batch
                  << " canais=" << (m.rgb ? "mag1c,460,550,640" : "mag1c,640,550,460") << '\n';
        if (o.modo == "all" || o.modo == "model_only")
        {
            fase = "model_only";
            executar(o, m, samples, prepared, "model_only");
        }
        if (o.modo == "all" || o.modo == "end_to_end")
        {
            fase = "end_to_end";
            executar(o, m, samples, prepared, "end_to_end");
        }
        prepared.clear();
        prepared.shrink_to_fit();
        fase = "validacao";
        temporarios = o.saida / ".scores_pr";
        fs::create_directory(temporarios);
        std::vector<fs::path> runs;
        std::array<std::vector<ResultadoImagem>, 2> rows;
        std::array<ResumoMetricas, 2> totals;
        std::vector<float> input(4 * PIXELS), logits(PIXELS);
        std::vector<std::uint8_t> mask(PIXELS);
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const auto &a = samples[i];
            preparar(carregar_canais(a, m.rgb), input, m.patch);
            m.inferir(input);
            reconstruir_patches(m.patches_saida.data(), logits.data(), m.patch, sizeof(float));
            const auto label = carregar_label(a);
            posprocessar(logits.data(), logits.size() * sizeof(float), true, mask);
            std::vector<PixelPR> pixels;
            pixels.reserve(PIXELS);
            for (int y = 0; y < 512; ++y)
                for (int x = 0; x < 512; ++x)
                {
                    const float truth = label.ptr<float>(y)[x], z = logits[y * 512 + x];
                    if (truth != 0 && truth != 1)
                        throw std::runtime_error("Label deve ser binario: " + a.id);
                    // Probabilidade FLOAT32: respeita empates do sigmoid saturado.
                    const float score = 1.0f / (1.0f + std::exp(-z));
                    pixels.push_back({score, truth != 0});
                }
            runs.push_back(temporarios / (std::to_string(i) + ".bin"));
            const double ap = salvar_scores(pixels, runs.back());
            for (int official = 0; official < 2; ++official)
            {
                if (official)
                    abrir_mascara(mask);
                auto r = medir_imagem(a, mask, label, ap, official);
                totals[official].adicionar(r);
                rows[official].push_back(std::move(r));
            }
            mostrar_progresso("validacao", i + 1, samples.size());
        }
        fase = "AUPRC global";
        const double auc = auprc_global(runs, temporarios);
        salvar_metricas(o.saida, o, m, rows[0], totals[0], false, auc);
        salvar_metricas(o.saida / "validacao_oficial", o, m, rows[1], totals[1], true, auc);
        fs::remove_all(temporarios);
        temporarios.clear();
        config(o, m, samples.size(), model_hash, csv_hash, "concluido");
        std::cout << "Resultados: " << fs::absolute(o.saida) << '\n';
        return 0;
    }
    catch (const std::exception &e)
    {
        if (!temporarios.empty())
        {
            std::error_code ec;
            fs::remove_all(temporarios, ec);
        }
        if (!output.empty())
        {
            std::ofstream failed(output / "falha.txt");
            failed << fase << ": " << e.what() << '\n';
        }
        std::cerr << "benchmark_arm (" << fase << "): " << e.what() << '\n';
        return 1;
    }
}

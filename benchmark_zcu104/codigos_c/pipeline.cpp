#include "pipeline.hpp"

#include "postprocess.hpp"
#include "power.hpp"
#include "preprocess.hpp"
#include "xmodel_runner.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point inicio, Clock::time_point fim) {
    return std::chrono::duration<double, std::milli>(fim - inicio).count();
}

double segundos(Clock::time_point inicio, Clock::time_point fim) {
    return std::chrono::duration<double>(fim - inicio).count();
}

void validar(const std::vector<Amostra>& amostras, const ConfiguracaoPipeline& c) {
    if (amostras.empty() || c.runners < 1 || c.runners > 4 ||
        c.nucleos_cpu < 1 || c.nucleos_cpu > 4 ||
        c.workers_pre < 1 || c.workers_pre > 4 ||
        c.workers_pos < 1 || c.workers_pos > 4 ||
        c.slots_por_runner < 1 || c.slots_por_runner > 4 || c.warmup < 0)
        throw std::runtime_error("Configuracao do pipeline invalida");
}

std::size_t trabalhos(const std::vector<Amostra>& amostras,
                     const ConfiguracaoPipeline& c) {
    return c.inferencias == 0 ? amostras.size() : c.inferencias;
}

class AfinidadeCpu {
public:
    explicit AfinidadeCpu(int quantidade) {
        if (pthread_getaffinity_np(pthread_self(), sizeof(original_), &original_) != 0)
            throw std::runtime_error("Nao foi possivel ler afinidade da CPU");
        for (int cpu = 0; cpu < CPU_SETSIZE &&
                          static_cast<int>(cpus_.size()) < quantidade; ++cpu)
            if (CPU_ISSET(cpu, &original_)) cpus_.push_back(cpu);
        if (static_cast<int>(cpus_.size()) < quantidade)
            throw std::runtime_error("Nucleos de CPU insuficientes");

        cpu_set_t limite;
        CPU_ZERO(&limite);
        for (int cpu : cpus_) CPU_SET(cpu, &limite);
        if (pthread_setaffinity_np(pthread_self(), sizeof(limite), &limite) != 0)
            throw std::runtime_error("Nao foi possivel limitar a CPU");
    }

    ~AfinidadeCpu() {
        pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_);
    }

    void fixar_thread(int indice) const {
        cpu_set_t unico;
        CPU_ZERO(&unico);
        CPU_SET(cpus_[static_cast<std::size_t>(indice) % cpus_.size()], &unico);
        if (pthread_setaffinity_np(pthread_self(), sizeof(unico), &unico) != 0)
            throw std::runtime_error("Nao foi possivel fixar thread na CPU");
    }

private:
    cpu_set_t original_{};
    std::vector<int> cpus_;
};

template <typename T>
class Fila {
public:
    explicit Fila(std::size_t capacidade) : capacidade_(capacidade) {}

    bool colocar(T valor) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return fechada_ || dados_.size() < capacidade_; });
        if (fechada_) return false;
        dados_.push_back(std::move(valor));
        cv_.notify_all();
        return true;
    }

    bool retirar(T& valor) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return fechada_ || !dados_.empty(); });
        if (fechada_) return false;
        valor = std::move(dados_.front());
        dados_.pop_front();
        cv_.notify_all();
        return true;
    }

    void fechar() {
        std::lock_guard<std::mutex> lock(mutex_);
        fechada_ = true;
        cv_.notify_all();
    }

private:
    const std::size_t capacidade_;
    std::deque<T> dados_;
    bool fechada_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class Controle {
public:
    explicit Controle(std::size_t total) : total_(total) {}

    void concluir(Clock::time_point instante) {
        if (++concluidas_ == total_) {
            std::lock_guard<std::mutex> lock(mutex_);
            fim_ = instante;
            finalizado_ = true;
            cv_.notify_all();
        }
    }

    void falhar(std::exception_ptr erro) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!erro_) erro_ = erro;
        cv_.notify_all();
    }

    Clock::time_point esperar() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return erro_ || finalizado_; });
        return fim_;
    }

    void relancar_erro() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (erro_) std::rethrow_exception(erro_);
    }

    std::size_t concluidas() const { return concluidas_; }

private:
    const std::size_t total_;
    std::atomic<std::size_t> concluidas_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::exception_ptr erro_;
    Clock::time_point fim_{};
    bool finalizado_ = false;
};

void juntar(std::vector<std::thread>& threads) {
    for (auto& thread : threads) if (thread.joinable()) thread.join();
}

void preparar_slots(std::vector<XModelRunner>& runners,
                    const std::vector<Amostra>& amostras,
                    int warmup) {
    for (std::size_t r = 0; r < runners.size(); ++r) {
        auto& runner = runners[r];
        for (std::size_t s = 0; s < runner.slots.size(); ++s) {
            auto canais = carregar_canais(
                amostras[(r * runner.slots.size() + s) % amostras.size()]);
            auto& slot = *runner.slots[s];
            preprocessar(canais, slot.dados_entrada(), slot.bytes_entrada(),
                         runner.escala_entrada);
            runner.sincronizar_entrada(slot);
        }
        for (int i = 0; i < warmup; ++i)
            runner.inferir(*runner.slots[static_cast<std::size_t>(i) %
                                        runner.slots.size()]);
    }
}

struct Trabalho {
    XModelRunner* runner;
    SlotXModel* slot;
    std::size_t lane;
    std::vector<std::uint8_t> mascara = std::vector<std::uint8_t>(PIXELS_SAIDA);
    TemposImagem tempos;
    Clock::time_point inicio;
    Clock::time_point enfileirado_runner;
    Clock::time_point enfileirado_pos;
};

} // namespace

ResultadoExecucao executar_model_only(
    const std::string& caminho_xmodel,
    const std::vector<Amostra>& amostras,
    const ConfiguracaoPipeline& configuracao,
    MonitorPotencia* potencia,
    std::atomic<std::size_t>* progresso) {
    validar(amostras, configuracao);
    AfinidadeCpu afinidade(configuracao.nucleos_cpu);
    auto runners = carregar_runners(caminho_xmodel, configuracao.runners,
                                    configuracao.slots_por_runner);
    preparar_slots(runners, amostras, configuracao.warmup);

    const std::size_t total = trabalhos(amostras, configuracao);
    ResultadoExecucao resultado;
    resultado.modo = "model_only";
    resultado.configuracao = configuracao;
    resultado.imagens.resize(total);
    Controle controle(total);
    std::atomic<std::size_t> proximo{0};
    std::vector<std::thread> threads;
    threads.reserve(runners.size());

    if (potencia) potencia->iniciar();
    const auto inicio = Clock::now();
    try {
        for (std::size_t r = 0; r < runners.size(); ++r) {
            threads.emplace_back([&, r] {
                try {
                    if (configuracao.fixar_afinidade) afinidade.fixar_thread(r);
                    auto& runner = runners[r];
                    std::size_t local = 0;
                    while (true) {
                        const std::size_t job = proximo.fetch_add(1);
                        if (job >= total) break;
                        const std::size_t s = local++ % runner.slots.size();
                        auto& slot = *runner.slots[s];
                        const auto t0 = Clock::now();
                        runner.inferir(slot);
                        const auto t1 = Clock::now();
                        auto& t = resultado.imagens[job];
                        t.trabalho = job;
                        t.indice_amostra =
                            (r * runner.slots.size() + s) % amostras.size();
                        t.inferencia_ms = ms(t0, t1);
                        t.latencia_total_ms = t.inferencia_ms;
                        if (progresso) progresso->fetch_add(1, std::memory_order_relaxed);
                        controle.concluir(t1);
                    }
                } catch (...) {
                    controle.falhar(std::current_exception());
                }
            });
        }
    } catch (...) {
        proximo = total;
        juntar(threads);
        if (potencia) potencia->parar();
        throw;
    }

    const auto fim = controle.esperar();
    if (potencia) potencia->parar();
    juntar(threads);
    controle.relancar_erro();

    resultado.concluidas = controle.concluidas();
    resultado.duracao_s = segundos(inicio, fim);
    resultado.throughput_fps = resultado.concluidas / resultado.duracao_s;
    return resultado;
}

ResultadoExecucao executar_end_to_end(
    const std::string& caminho_xmodel,
    const std::vector<Amostra>& amostras,
    const ConfiguracaoPipeline& configuracao,
    MonitorPotencia* potencia,
    std::atomic<std::size_t>* progresso) {
    validar(amostras, configuracao);
    AfinidadeCpu afinidade(configuracao.nucleos_cpu);
    auto runners = carregar_runners(caminho_xmodel, configuracao.runners,
                                    configuracao.slots_por_runner);
    preparar_slots(runners, amostras, configuracao.warmup);

    const std::size_t total = trabalhos(amostras, configuracao);
    const std::size_t capacidade =
        static_cast<std::size_t>(configuracao.runners * configuracao.slots_por_runner);
    ResultadoExecucao resultado;
    resultado.modo = "end_to_end";
    resultado.configuracao = configuracao;
    resultado.imagens.resize(total);
    Controle controle(total);
    std::atomic<std::size_t> proximo{0};

    Fila<Trabalho*> livres(capacidade);
    Fila<Trabalho*> pos(capacidade);
    std::vector<std::unique_ptr<Fila<Trabalho*>>> filas_runner;
    std::vector<std::unique_ptr<Trabalho>> slots;
    for (std::size_t r = 0; r < runners.size(); ++r) {
        filas_runner.push_back(std::make_unique<Fila<Trabalho*>>(
            configuracao.slots_por_runner));
        for (auto& slot : runners[r].slots) {
            auto trabalho = std::make_unique<Trabalho>();
            trabalho->runner = &runners[r];
            trabalho->slot = slot.get();
            trabalho->lane = r;
            livres.colocar(trabalho.get());
            slots.push_back(std::move(trabalho));
        }
    }

    auto fechar_filas = [&] {
        livres.fechar();
        pos.fechar();
        for (auto& fila : filas_runner) fila->fechar();
    };
    auto falhar = [&] {
        controle.falhar(std::current_exception());
        fechar_filas();
    };

    std::vector<std::thread> threads;
    threads.reserve(configuracao.workers_pre + configuracao.runners +
                    configuracao.workers_pos);
    if (potencia) potencia->iniciar();
    const auto inicio = Clock::now();
    try {
        for (int p = 0; p < configuracao.workers_pos; ++p) {
            threads.emplace_back([&, p] {
                try {
                    if (configuracao.fixar_afinidade) afinidade.fixar_thread(p);
                    Trabalho* w;
                    while (pos.retirar(w)) {
                        const auto t0 = Clock::now();
                        w->tempos.espera_pos_ms = ms(w->enfileirado_pos, t0);
                        posprocessar(w->slot->dados_saida(), w->slot->bytes_saida(),
                                     w->runner->saida_float, w->mascara);
                        const auto t1 = Clock::now();
                        w->tempos.postprocess_ms = ms(t0, t1);
                        w->tempos.latencia_total_ms = ms(w->inicio, t1);
                        resultado.imagens[w->tempos.trabalho] = w->tempos;
                        if (progresso) progresso->fetch_add(1, std::memory_order_relaxed);
                        controle.concluir(t1);
                        if (!livres.colocar(w)) break;
                    }
                } catch (...) { falhar(); }
            });
        }
        for (std::size_t r = 0; r < runners.size(); ++r) {
            threads.emplace_back([&, r] {
                try {
                    if (configuracao.fixar_afinidade)
                        afinidade.fixar_thread(configuracao.workers_pos + r);
                    Trabalho* w;
                    while (filas_runner[r]->retirar(w)) {
                        auto t0 = Clock::now();
                        w->tempos.espera_runner_ms = ms(w->enfileirado_runner, t0);
                        w->runner->sincronizar_entrada(*w->slot);
                        auto t1 = Clock::now();
                        w->tempos.sync_entrada_ms = ms(t0, t1);
                        w->runner->inferir(*w->slot);
                        auto t2 = Clock::now();
                        w->tempos.inferencia_ms = ms(t1, t2);
                        w->runner->sincronizar_saida(*w->slot);
                        auto t3 = Clock::now();
                        w->tempos.sync_saida_ms = ms(t2, t3);
                        w->enfileirado_pos = t3;
                        if (!pos.colocar(w)) break;
                    }
                } catch (...) { falhar(); }
            });
        }
        for (int p = 0; p < configuracao.workers_pre; ++p) {
            threads.emplace_back([&, p] {
                try {
                    if (configuracao.fixar_afinidade)
                        afinidade.fixar_thread(configuracao.workers_pos +
                                               configuracao.runners + p);
                    while (true) {
                        const std::size_t job = proximo.fetch_add(1);
                        if (job >= total) break;
                        const auto t0 = Clock::now();
                        Trabalho* w;
                        if (!livres.retirar(w)) break;
                        const auto t1 = Clock::now();
                        w->inicio = t0;
                        w->tempos = {};
                        w->tempos.trabalho = job;
                        w->tempos.indice_amostra = job % amostras.size();
                        w->tempos.espera_slot_ms = ms(t0, t1);
                        auto canais = carregar_canais(amostras[w->tempos.indice_amostra]);
                        const auto t2 = Clock::now();
                        w->tempos.leitura_ms = ms(t1, t2);
                        preprocessar(canais, w->slot->dados_entrada(),
                                     w->slot->bytes_entrada(),
                                     w->runner->escala_entrada);
                        const auto t3 = Clock::now();
                        w->tempos.preprocess_ms = ms(t2, t3);
                        w->enfileirado_runner = t3;
                        if (!filas_runner[w->lane]->colocar(w)) break;
                    }
                } catch (...) { falhar(); }
            });
        }
    } catch (...) {
        fechar_filas();
        juntar(threads);
        if (potencia) potencia->parar();
        throw;
    }

    const auto fim = controle.esperar();
    if (potencia) potencia->parar();
    fechar_filas();
    juntar(threads);
    controle.relancar_erro();

    resultado.concluidas = controle.concluidas();
    resultado.duracao_s = segundos(inicio, fim);
    resultado.throughput_fps = resultado.concluidas / resultado.duracao_s;
    return resultado;
}

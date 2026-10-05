#include "power.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

double segundos(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

std::string ler_linha(const fs::path& arquivo) {
    std::ifstream entrada(arquivo);
    std::string texto;
    std::getline(entrada, texto);
    return texto;
}

} // namespace

MonitorPotencia::MonitorPotencia(int intervalo_ms) : intervalo_ms_(intervalo_ms) {
    if (intervalo_ms <= 0) throw std::runtime_error("Intervalo de potencia invalido");

    const fs::path raiz("/sys/class/hwmon");
    if (!fs::exists(raiz)) return;

    for (const auto& hwmon : fs::directory_iterator(raiz)) {
        const fs::path fonte_name = hwmon.path() / "name";
        const std::string chip = ler_linha(fonte_name);
        for (const auto& entrada : fs::directory_iterator(hwmon.path())) {
            const std::string nome = entrada.path().filename().string();
            if (nome.rfind("power", 0) != 0 || nome.size() < 12 ||
                nome.compare(nome.size() - 6, 6, "_input") != 0) continue;

            const std::string id = nome.substr(0, nome.size() - 6);
            const fs::path fonte_label = hwmon.path() / (id + "_label");
            const std::string label = ler_linha(fonte_label);
            trilhos_.push_back({label.empty() ? chip + ":" + id : label,
                                chip, fonte_name.string(),
                                fs::exists(fonte_label) ? fonte_label.string() : "",
                                entrada.path(), {}});
        }
    }
}

MonitorPotencia::~MonitorPotencia() { parar(); }

bool MonitorPotencia::disponivel() const { return !trilhos_.empty(); }

void MonitorPotencia::amostrar() {
    for (auto& trilho : trilhos_) {
        std::ifstream entrada(trilho.arquivo);
        double microwatts;
        if (entrada >> microwatts && std::isfinite(microwatts) && microwatts >= 0)
            trilho.leituras.emplace_back(segundos(std::chrono::steady_clock::now()),
                                         microwatts / 1'000'000.0);
    }
}

void MonitorPotencia::iniciar() {
    if (thread_.joinable()) throw std::runtime_error("Monitor de potencia ja iniciado");
    if (trilhos_.empty()) return;
    for (auto& trilho : trilhos_) trilho.leituras.clear();
    parar_ = false;
    amostrar();
    thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!cv_.wait_for(lock, std::chrono::milliseconds(intervalo_ms_),
                             [this] { return parar_.load(); })) {
            lock.unlock();
            amostrar();
            lock.lock();
        }
    });
}

void MonitorPotencia::parar() {
    if (!thread_.joinable()) return;
    { std::lock_guard<std::mutex> lock(mutex_); parar_ = true; }
    cv_.notify_all();
    thread_.join();
    amostrar();
}

double integrar_potencia(const std::vector<std::pair<double, double>>& leituras,
                        double inicio_s, double fim_s) {
    if (fim_s < inicio_s) throw std::runtime_error("Duracao negativa");
    if (leituras.empty()) return 0;
    double energia = 0;
    // Fora das leituras usa o valor de borda; dentro interpola linearmente.
    if (inicio_s < leituras.front().first)
        energia += (std::min(fim_s, leituras.front().first) - inicio_s) * leituras.front().second;
    for (std::size_t i = 1; i < leituras.size(); ++i) {
        const auto [t0, w0] = leituras[i - 1];
        const auto [t1, w1] = leituras[i];
        const double a = std::max(inicio_s, t0), b = std::min(fim_s, t1);
        if (b <= a || t1 <= t0) continue;
        const double wa = w0 + (w1 - w0) * (a - t0) / (t1 - t0);
        const double wb = w0 + (w1 - w0) * (b - t0) / (t1 - t0);
        energia += (b - a) * (wa + wb) / 2;
    }
    if (fim_s > leituras.back().first)
        energia += (fim_s - std::max(inicio_s, leituras.back().first)) * leituras.back().second;
    return energia;
}

std::vector<MedidaPotencia> MonitorPotencia::resumo(
    std::chrono::steady_clock::time_point inicio,
    std::chrono::steady_clock::time_point fim) const {
    if (thread_.joinable()) throw std::runtime_error("Pare o monitor antes do resumo");
    const double a = segundos(inicio), b = segundos(fim);
    if (b < a) throw std::runtime_error("Duracao negativa");
    std::vector<MedidaPotencia> resultado;
    for (const auto& trilho : trilhos_) {
        if (trilho.leituras.empty()) continue;
        const auto limites = std::minmax_element(trilho.leituras.begin(), trilho.leituras.end(),
            [](const auto& x, const auto& y) { return x.second < y.second; });
        const double energia = integrar_potencia(trilho.leituras, a, b);
        resultado.push_back({trilho.nome, trilho.sensor_chip,
                             trilho.fonte_name, trilho.fonte_label,
                             trilho.arquivo.string(), trilho.leituras.size(),
                             b > a ? energia / (b - a) : 0.0,
                             limites.first->second, limites.second->second, energia});
    }
    return resultado;
}

#include "power.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

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
        const std::string chip = ler_linha(hwmon.path() / "name");
        for (const auto& entrada : fs::directory_iterator(hwmon.path())) {
            const std::string nome = entrada.path().filename().string();
            if (nome.rfind("power", 0) != 0 || nome.size() < 12 ||
                nome.compare(nome.size() - 6, 6, "_input") != 0) continue;

            const std::string id = nome.substr(0, nome.size() - 6);
            const std::string label = ler_linha(hwmon.path() / (id + "_label"));
            trilhos_.push_back({label.empty() ? chip + ":" + id : label,
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
            trilho.watts.push_back(microwatts / 1'000'000.0);
    }
}

void MonitorPotencia::iniciar() {
    if (thread_.joinable()) throw std::runtime_error("Monitor de potencia ja iniciado");
    if (trilhos_.empty()) return;
    for (auto& trilho : trilhos_) trilho.watts.clear();
    parar_ = false;
    amostrar();
    thread_ = std::thread([this] {
        while (!parar_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalo_ms_));
            if (!parar_) amostrar();
        }
    });
}

void MonitorPotencia::parar() {
    if (!thread_.joinable()) return;
    parar_ = true;
    thread_.join();
    amostrar();
}

std::vector<MedidaPotencia> MonitorPotencia::resumo(double duracao_s) const {
    if (thread_.joinable()) throw std::runtime_error("Pare o monitor antes do resumo");
    if (duracao_s < 0) throw std::runtime_error("Duracao negativa");

    std::vector<MedidaPotencia> resultado;
    for (const auto& trilho : trilhos_) {
        if (trilho.watts.empty()) continue;
        const auto limites = std::minmax_element(trilho.watts.begin(), trilho.watts.end());
        const double soma = std::accumulate(trilho.watts.begin(), trilho.watts.end(), 0.0);
        const double media = soma / trilho.watts.size();
        resultado.push_back({trilho.nome, trilho.watts.size(), media,
                             *limites.first, *limites.second, media * duracao_s});
    }
    return resultado;
}

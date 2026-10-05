#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <cstddef>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

struct MedidaPotencia {
    std::string trilho;
    std::string sensor_chip;
    std::string fonte_name;
    std::string fonte_label;
    std::string fonte_power_input;
    std::size_t amostras = 0;
    double media_w = 0;
    double minima_w = 0;
    double maxima_w = 0;
    double energia_j = 0;
};

// Pares (tempo monotonic em segundos, watts), ordenados pelo tempo.
double integrar_potencia(const std::vector<std::pair<double, double>>& leituras,
                        double inicio_s, double fim_s);

class MonitorPotencia {
public:
    explicit MonitorPotencia(int intervalo_ms = 200);
    ~MonitorPotencia();

    void iniciar();
    void parar();
    std::vector<MedidaPotencia> resumo(std::chrono::steady_clock::time_point inicio,
                                       std::chrono::steady_clock::time_point fim) const;
    bool disponivel() const;

private:
    struct Trilho {
        std::string nome;
        std::string sensor_chip;
        std::string fonte_name;
        std::string fonte_label;
        std::filesystem::path arquivo;
        std::vector<std::pair<double, double>> leituras;
    };

    void amostrar();

    int intervalo_ms_;
    std::atomic<bool> parar_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<Trilho> trilhos_;
};

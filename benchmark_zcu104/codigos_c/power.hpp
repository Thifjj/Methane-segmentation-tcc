#pragma once

#include <atomic>
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

class MonitorPotencia {
public:
    explicit MonitorPotencia(int intervalo_ms = 200);
    ~MonitorPotencia();

    void iniciar();
    void parar();
    std::vector<MedidaPotencia> resumo(double duracao_s) const;
    bool disponivel() const;

private:
    struct Trilho {
        std::string nome;
        std::string sensor_chip;
        std::string fonte_name;
        std::string fonte_label;
        std::filesystem::path arquivo;
        std::vector<double> watts;
    };

    void amostrar();

    int intervalo_ms_;
    std::atomic<bool> parar_{false};
    std::thread thread_;
    std::vector<Trilho> trilhos_;
};

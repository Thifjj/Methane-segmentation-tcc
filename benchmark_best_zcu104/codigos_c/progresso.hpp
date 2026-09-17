#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <limits>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

class Progresso {
public:
    Progresso(std::string etapa, std::size_t total, std::ostream& saida = std::cout)
        : etapa_(std::move(etapa)), total_(total), saida_(saida),
          thread_([this] { relatar(); }) {}

    ~Progresso() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            parar_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    Progresso(const Progresso&) = delete;
    Progresso& operator=(const Progresso&) = delete;

    std::atomic<std::size_t>* contador() { return &concluidas_; }

private:
    void relatar() {
        std::size_t ultimo = std::numeric_limits<std::size_t>::max();
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            const auto atual = concluidas_.load(std::memory_order_relaxed);
            if (atual != ultimo) {
                saida_ << etapa_ << ": " << atual << "/" << total_ << std::endl;
                ultimo = atual;
            }
            if (parar_) break;
            cv_.wait_for(lock, std::chrono::seconds(1), [this] { return parar_; });
        }
    }

    std::string etapa_;
    std::size_t total_;
    std::ostream& saida_;
    std::atomic<std::size_t> concluidas_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    bool parar_ = false;
    std::thread thread_;
};

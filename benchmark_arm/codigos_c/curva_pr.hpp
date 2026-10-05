#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

// FP32 scores reais: nenhuma re-quantização INT8 ou aproximação por bins.
struct ScorePR { float score; std::uint64_t positivos, negativos; };
struct PixelPR { float score; bool positivo; };

class CurvaPR {
public:
    void adicionar(const ScorePR& s) {
        tp_ += s.positivos; fp_ += s.negativos;
        const double precision = double(tp_) / (tp_ + fp_);
        ap_numerador_ += s.positivos * precision;
        auc_numerador_ += s.positivos * (precision_anterior_ + precision) / 2;
        precision_anterior_ = precision;
    }
    double ap() const { return tp_ ? ap_numerador_ / tp_ : std::numeric_limits<double>::quiet_NaN(); }
    double auc() const { return tp_ ? auc_numerador_ / tp_ : 0.5; }
private:
    std::uint64_t tp_ = 0, fp_ = 0;
    double precision_anterior_ = 1, ap_numerador_ = 0, auc_numerador_ = 0;
};

inline double salvar_scores(std::vector<PixelPR>& pixels, const std::filesystem::path& path) {
    std::sort(pixels.begin(), pixels.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Falha ao criar scores PR: " + path.string());
    CurvaPR curva;
    for (std::size_t i = 0; i < pixels.size();) {
        ScorePR s{pixels[i].score, 0, 0};
        do {
            if (pixels[i].positivo) ++s.positivos; else ++s.negativos;
            ++i;
        } while (i < pixels.size() && pixels[i].score == s.score);
        curva.adicionar(s);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    if (!f) throw std::runtime_error("Disco insuficiente ao salvar scores PR");
    return curva.ap();
}

inline bool ler_score(std::ifstream& f, ScorePR& s) {
    f.read(reinterpret_cast<char*>(&s), sizeof(s));
    if (f.gcount() == sizeof(s)) return true;
    if (f.gcount() || !f.eof()) throw std::runtime_error("Arquivo PR incompleto");
    return false;
}

inline double mesclar_scores(const std::vector<std::filesystem::path>& paths,
                            const std::filesystem::path& destino = {}) {
    struct Item { ScorePR score; std::size_t arquivo; };
    const auto menor = [](const Item& a, const Item& b) { return a.score.score < b.score.score; };
    std::priority_queue<Item, std::vector<Item>, decltype(menor)> fila(menor);
    std::vector<std::ifstream> arquivos;
    arquivos.reserve(paths.size());
    for (const auto& p : paths) {
        arquivos.emplace_back(p, std::ios::binary);
        if (!arquivos.back()) throw std::runtime_error("Falha ao abrir scores PR");
        ScorePR s{};
        if (ler_score(arquivos.back(), s)) fila.push({s, arquivos.size()-1});
    }
    std::ofstream out;
    if (!destino.empty()) {
        out.open(destino, std::ios::binary);
        if (!out) throw std::runtime_error("Falha ao criar merge PR");
    }
    CurvaPR curva;
    while (!fila.empty()) {
        ScorePR s{fila.top().score.score, 0, 0};
        do {
            const auto item = fila.top(); fila.pop();
            s.positivos += item.score.positivos; s.negativos += item.score.negativos;
            ScorePR proximo{};
            if (ler_score(arquivos[item.arquivo], proximo)) fila.push({proximo, item.arquivo});
        } while (!fila.empty() && fila.top().score.score == s.score);
        curva.adicionar(s);
        if (out.is_open()) out.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    if (out.is_open() && !out) throw std::runtime_error("Disco insuficiente no merge PR");
    return curva.auc();
}

// Fan-in limitado para FULL: no máximo 32 arquivos abertos, memória por imagem.
inline double auprc_global(std::vector<std::filesystem::path> paths, const std::filesystem::path& tmp) {
    std::size_t nivel = 0;
    while (paths.size() > 32) {
        std::vector<std::filesystem::path> proximos;
        for (std::size_t i = 0; i < paths.size(); i += 32) {
            const auto fim = std::min(paths.size(), i+32);
            const auto destino = tmp / ("merge_"+std::to_string(nivel)+"_"+std::to_string(i)+".bin");
            mesclar_scores({paths.begin()+i, paths.begin()+fim}, destino);
            for (std::size_t j = i; j < fim; ++j) std::filesystem::remove(paths[j]);
            proximos.push_back(destino);
        }
        paths = std::move(proximos); ++nivel;
    }
    return mesclar_scores(paths);
}

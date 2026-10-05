#pragma once
#include "dataset.hpp"
#include "postprocess.hpp"
#include "curva_pr.hpp"
#include <array>
#include <numeric>

inline double razao(double a, double b) { return b ? a/b : 0.0; }
struct Contagens {
    std::uint64_t tp=0, fp=0, fn=0, tn=0;
    void somar(const Contagens& c) { tp+=c.tp;fp+=c.fp;fn+=c.fn;tn+=c.tn; }
};
inline std::array<double,6> metricas(const Contagens& c, bool oficial=false) {
    const double eps=oficial?1e-6:0;
    return {razao(c.tp,c.tp+c.fp),razao(c.tp,c.tp+c.fn),
        razao(2.0*c.tp,2.0*c.tp+c.fp+c.fn+eps),razao(c.tp,c.tp+c.fp+c.fn+eps),
        razao(c.fp,c.fp+c.tn),razao(c.tp+c.tn,c.tp+c.fp+c.fn+c.tn)};
}
struct ResultadoImagem {
    std::string id, dificuldade, difficulty_csv;
    bool positive=false;
    Contagens c;
    std::size_t pixels_preditos=0;
    double average_precision=std::numeric_limits<double>::quiet_NaN();
};
inline ResultadoImagem medir_imagem(const Amostra& a, const std::vector<std::uint8_t>& mask,
                                    const cv::Mat& label, double ap, bool oficial) {
    ResultadoImagem r;
    r.id=a.id;r.difficulty_csv=a.difficulty;r.average_precision=ap;
    for(int y=0;y<512;++y)for(int x=0;x<512;++x){
        const float truth=label.ptr<float>(y)[x];
        if(truth!=0.0f && truth!=1.0f) throw std::runtime_error("Label deve ser binario e finito: "+a.id);
        const bool pred=mask[y*512+x]!=0, real=truth!=0;
        r.pixels_preditos+=pred;
        if(pred&&real)++r.c.tp;else if(pred)++r.c.fp;else if(real)++r.c.fn;else ++r.c.tn;
    }
    r.positive=r.c.tp+r.c.fn>0;
    const bool background=oficial?!r.positive:!a.has_plume;
    const bool strong=oficial?a.difficulty=="easy":a.qplume>1000;
    r.dificuldade=background?"sem_pluma":strong?"forte":"fraca";
    return r;
}
struct ResumoMetricas {
    Contagens global,forte,fraca,sem_pluma;
    std::size_t imagens=0,imagens_positivas=0,fp_tiles=0,tn_tiles=0;
    double soma_ap=0;
    void adicionar(const ResultadoImagem& r) {
        ++imagens;global.somar(r.c);
        if(r.positive){++imagens_positivas;soma_ap+=r.average_precision;}
        if(r.dificuldade=="sem_pluma"){
            sem_pluma.somar(r.c);
            if(r.pixels_preditos>640)++fp_tiles;else ++tn_tiles;
        } else if(r.dificuldade=="forte")forte.somar(r.c);else fraca.somar(r.c);
    }
};

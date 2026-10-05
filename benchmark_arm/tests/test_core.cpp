#include "../codigos_c/curva_pr.hpp"
#include "../codigos_c/sha256.hpp"
#include "../codigos_c/power.hpp"
#include <cassert>
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    const std::filesystem::path tmp(argv[1]);std::filesystem::create_directories(tmp);
    // Known vector, empty message; pad boundaries are checked independently in Python integration.
    std::ofstream(tmp/"empty").close();
    assert(arquivo_sha256(tmp/"empty")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    {std::ofstream f(tmp/"abc");f<<"abc";}
    assert(arquivo_sha256(tmp/"abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // Scores 0.9:+, 0.7:-, 0.5:+ produce AP=5/6, trapezoidal PR=19/24.
    std::vector<PixelPR> p{{.9f,true},{.7f,false},{.5f,true}};
    assert(std::abs(salvar_scores(p,tmp/"pr")-5.0/6)<1e-12);
    assert(std::abs(mesclar_scores({tmp/"pr"})-19.0/24)<1e-12);
    // Cross-file ties aggregate before integration, independently of run boundaries.
    std::vector<PixelPR> a{{.5f,true}},b{{.5f,false}};
    salvar_scores(a,tmp/"a");salvar_scores(b,tmp/"b");
    assert(std::abs(mesclar_scores({tmp/"a",tmp/"b"})-.75)<1e-12);
    assert(std::abs(integrar_potencia({{0,10},{2,20}},.5,1.5)-15)<1e-12);
    assert(std::abs(integrar_potencia({{0,10},{2,20}},-1,3)-60)<1e-12);
    std::cout<<"Core OK: SHA256, AP/PR com empates e energia trapezoidal.\n";
}

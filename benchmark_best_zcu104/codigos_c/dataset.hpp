  #pragma once

  #include <array>
  #include <cstddef>
  #include <filesystem>
  #include <optional>
  #include <string>
  #include <vector>
  #include <opencv2/core.hpp>

  struct Amostra{
    std::string id;
    std::filesystem::path pasta;
    std::optional<cv::Rect> janela;
    double qplume = 0.0;

  };

  using CanaisEntrada = std::array<cv::Mat,4>;

  std::vector<Amostra> carregar_amostras(const std::filesystem::path& csv, const std::filesystem::path& raiz_dataset, std::size_t limite = 0);

  CanaisEntrada carregar_canais(const Amostra& amostra);

  cv::Mat carregar_label(const Amostra& amostra);
  

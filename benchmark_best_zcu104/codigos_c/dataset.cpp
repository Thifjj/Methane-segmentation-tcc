
  #include "dataset.hpp"

  #include <algorithm>
  #include <cmath>
  #include <fstream>
  #include <stdexcept>

  #include <opencv2/imgcodecs.hpp>

  namespace fs = std::filesystem;

  namespace {

  std::vector<std::string> separar_csv(const std::string& linha) {
      std::vector<std::string> campos(1);
      bool entre_aspas = false;

      for (std::size_t i = 0; i < linha.size(); ++i) {
          char c = linha[i];

          if (c == '"') {
              if (entre_aspas && i + 1 < linha.size() && linha[i + 1] == '"') {
                  campos.back() += '"';
                  ++i;
              } else {
                  entre_aspas = !entre_aspas;
              }
          } else if (c == ',' && !entre_aspas) {
              campos.emplace_back();
          } else {
              campos.back() += c;
          }
      }

      if (entre_aspas) {
          throw std::runtime_error("Aspas não fechadas no CSV");
      }
      return campos;
  }

  int coluna(const std::vector<std::string>& cabecalho, const std::string& nome) {
      auto it = std::find(cabecalho.begin(), cabecalho.end(), nome);
      return it == cabecalho.end() ? -1 : static_cast<int>(it - cabecalho.begin());
  }

  std::string campo(const std::vector<std::string>& linha, int indice) {
      if (indice < 0 || static_cast<std::size_t>(indice) >= linha.size()) {
          return "";
      }
      return linha[indice];
  }

  bool booleano(const std::string& valor) {
      if (valor == "true" || valor == "True" || valor == "TRUE" || valor == "1") {
          return true;
      }
      if (valor == "false" || valor == "False" || valor == "FALSE" || valor == "0") {
          return false;
      }
      throw std::runtime_error("Valor booleano inválido: " + valor);
  }

  cv::Mat ler_tiff(const Amostra& amostra, const char* nome) {
      fs::path caminho = amostra.pasta / nome;
      cv::Mat imagem = cv::imread(caminho.string(), cv::IMREAD_UNCHANGED);

      if (imagem.empty()) {
          throw std::runtime_error("Não foi possível ler: " + caminho.string());
      }
      if (imagem.channels() != 1) {
          throw std::runtime_error("TIFF deve ter um canal: " + caminho.string());
      }

      if (amostra.janela) {
          const cv::Rect& j = *amostra.janela;
          if (j.x < 0 || j.y < 0 || j.width <= 0 || j.height <= 0 ||
              j.x > imagem.cols || j.y > imagem.rows ||
              j.width > imagem.cols - j.x ||
              j.height > imagem.rows - j.y) {
              throw std::runtime_error("Janela fora do TIFF: " + caminho.string());
          }
          imagem = imagem(j);
      }

      if (imagem.cols != 512 || imagem.rows != 512) {
          throw std::runtime_error("Esperado recorte 512x512 em: " + caminho.string());
      }

      cv::Mat resultado;
      imagem.convertTo(resultado, CV_32F);
      return resultado;
  }

  } // namespace

  std::vector<Amostra> carregar_amostras(
      const fs::path& csv,
      const fs::path& raiz_dataset,
      std::size_t limite
  ) {
      std::ifstream arquivo(csv);
      if (!arquivo) {
          throw std::runtime_error("Não foi possível abrir CSV: " + csv.string());
      }

      std::string linha;
      if (!std::getline(arquivo, linha)) {
          throw std::runtime_error("CSV vazio: " + csv.string());
      }
      if (!linha.empty() && linha.back() == '\r') linha.pop_back();

      auto cabecalho = separar_csv(linha);
      if (!cabecalho.empty() && cabecalho[0].rfind("\xEF\xBB\xBF", 0) == 0) {
          cabecalho[0].erase(0, 3);
      }

      const int id_col = coluna(cabecalho, "id");
      const int pasta_col = coluna(cabecalho, "folder");
      const int has_plume_col = coluna(cabecalho, "has_plume");
      const int qplume_col = coluna(cabecalho, "qplume");

      const int x_col = coluna(cabecalho, "window_col_off");
      const int y_col = coluna(cabecalho, "window_row_off");
      const int w_col = coluna(cabecalho, "window_width");
      const int h_col = coluna(cabecalho, "window_height");

      if (id_col < 0 && pasta_col < 0) {
          throw std::runtime_error("CSV precisa ter coluna id ou folder");
      }
      if (has_plume_col < 0 || qplume_col < 0) {
          throw std::runtime_error("CSV precisa ter colunas has_plume e qplume");
      }

      const int colunas_janela =
          (x_col >= 0) + (y_col >= 0) + (w_col >= 0) + (h_col >= 0);
      if (colunas_janela != 0 && colunas_janela != 4) {
          throw std::runtime_error("CSV tem apenas parte das colunas da janela");
      }

      std::vector<Amostra> amostras;
      std::size_t numero_linha = 1;

      while (std::getline(arquivo, linha)) {
          ++numero_linha;
          if (!linha.empty() && linha.back() == '\r') linha.pop_back();
          if (linha.empty()) continue;

          try {
              auto campos = separar_csv(linha);
              Amostra amostra;
              amostra.id = campo(campos, id_col);

              fs::path pasta_csv = campo(campos, pasta_col);
              if (!pasta_csv.has_filename()) pasta_csv = pasta_csv.parent_path();

              fs::path por_pasta = raiz_dataset / pasta_csv.filename();
              if (pasta_col >= 0) {
                  // Segue benchmark_manual: usa folder e ignora pastas ausentes.
                  if (!fs::is_directory(por_pasta)) continue;
                  amostra.pasta = por_pasta;
              } else {
                  amostra.pasta = raiz_dataset / amostra.id;
                  if (!fs::is_directory(amostra.pasta))
                      throw std::runtime_error("Pasta da amostra não encontrada");
              }

              if (amostra.id.empty()) {
                  amostra.id = amostra.pasta.filename().string();
              }

              amostra.has_plume = booleano(campo(campos, has_plume_col));
              const std::string qplume = campo(campos, qplume_col);
              if (qplume.empty()) {
                  if (amostra.has_plume) {
                      throw std::runtime_error("qplume ausente para amostra com pluma");
                  }
              } else {
                  amostra.qplume = std::stod(qplume);
                  if (amostra.has_plume && !std::isfinite(amostra.qplume)) {
                      throw std::runtime_error("qplume inválido para amostra com pluma");
                  }
              }

              if (colunas_janela == 4) {
                  amostra.janela = cv::Rect(
                      std::stoi(campo(campos, x_col)),
                      std::stoi(campo(campos, y_col)),
                      std::stoi(campo(campos, w_col)),
                      std::stoi(campo(campos, h_col))
                  );
              }

              amostras.push_back(std::move(amostra));
              if (limite != 0 && amostras.size() >= limite) break;
          } catch (const std::exception& e) {
              throw std::runtime_error(
                  "CSV linha " + std::to_string(numero_linha) + ": " + e.what()
              );
          }
      }

      if (amostras.empty()) {
          throw std::runtime_error("Nenhuma amostra encontrada no CSV");
      }
      return amostras;
  }

  CanaisEntrada carregar_canais(const Amostra& amostra) {
      return {
          ler_tiff(amostra, "mag1c.tif"),
          ler_tiff(amostra, "TOA_AVIRIS_640nm.tif"),
          ler_tiff(amostra, "TOA_AVIRIS_550nm.tif"),
          ler_tiff(amostra, "TOA_AVIRIS_460nm.tif")
      };
  }

  cv::Mat carregar_label(const Amostra& amostra) {
      return ler_tiff(amostra, "labelbinary.tif");
  }

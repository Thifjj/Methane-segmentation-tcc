# Guia do projeto de segmentação de metano na ZCU104

Este projeto compara cinco redes de segmentação de plumas de metano usando imagens STARCOP. O fluxo começa nos modelos PyTorch, passa por quantização INT8 e compilação no Vitis AI e termina com execução e avaliação na DPU da ZCU104.

## 1. Como o projeto está organizado

```text
Dataset STARCOP
      │
      ▼
Treinamento PyTorch ──► checkpoints .pth
      │
      ▼
Inspector + quantização Vitis AI ──► *_int.xmodel
      │
      ▼
Compilação para DPUCZDX8G/ZCU104 ──► methane_*.xmodel
      │
      ▼
Benchmark na placa ──► desempenho + qualidade + energia
```

| Local | Função |
|---|---|
| `Modelos/` | Implementações PyTorch das cinco arquiteturas |
| `Modelos_treinados/` | Checkpoints `.pth` treinados |
| `Utils/DataLoader.py` | Leitura do STARCOP e normalização das bandas |
| `Treinos/` e `main.ipynb` | Treinamento e uso dos modelos em PyTorch |
| `Testes/Teste_Unet.py` | Avaliação original usada como referência das métricas |
| `VitisAI/` | Inspeção, quantização INT8 e compilação |
| `build/vitis_ai/` | Artefatos gerados pelo Vitis AI |
| `Benchmark_ZCU104/` | Benchmark C++17 executado na placa |
| `results_all_models/` | Resultados copiados da ZCU104 |

Os modelos disponíveis são `baseline`, `depth_reduced`, `skip_connections`, `mobilenet_v2` e `mobilenet_v3`. Todos recebem quatro canais — `mag1c`, vermelho, verde e azul — e produzem uma máscara binária. Cada amostra do dataset precisa dos arquivos `mag1c.tif`, `TOA_AVIRIS_640nm.tif`, `TOA_AVIRIS_550nm.tif`, `TOA_AVIRIS_460nm.tif` e `labelbinary.tif`. O CSV deve conter a pasta ou o identificador da amostra e a janela espacial.

## 2. Reproduzir os modelos para a placa

O treinamento é feito em PyTorch e grava os pesos em `Modelos_treinados/`. A quantização e a compilação são feitas no container CPU do Vitis AI 3.5:

```bash
docker start methane-vitis-ai-cpu
docker exec -it methane-vitis-ai-cpu bash
conda activate vitis-ai-pytorch
cd /workspace/VitisAI
```

Inspecione, calibre e exporte os cinco modelos. A calibração abaixo usa 100 imagens reais e reproduzíveis:

```bash
set -e
for MODEL in baseline depth_reduced skip_connections mobilenet_v2 mobilenet_v3; do
  python inspect_model.py \
    --model "$MODEL" \
    --target DPUCZDX8G_ISA1_B4096 \
    --output-dir ../build/vitis_ai/inspect

  python quantize_model.py \
    --model "$MODEL" --quant-mode calib \
    --csv /dataset_STARCOP/train.csv \
    --data-root /dataset_STARCOP \
    --subset-len 100 \
    --target DPUCZDX8G_ISA1_B4096 \
    --output-dir ../build/vitis_ai/quantize

  python quantize_model.py \
    --model "$MODEL" --quant-mode test \
    --csv /dataset_STARCOP/train.csv \
    --data-root /dataset_STARCOP \
    --target DPUCZDX8G_ISA1_B4096 \
    --output-dir ../build/vitis_ai/quantize \
    --deploy
done
```

Compile usando o `arch.json` correspondente exatamente à DPU instalada na placa:

```bash
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
test -f "$ARCH" || { echo "arch.json ausente"; exit 1; }

for MODEL in baseline depth_reduced skip_connections mobilenet_v2 mobilenet_v3; do
  XMODEL="$(find "../build/vitis_ai/quantize/$MODEL" -maxdepth 1 -name '*_int.xmodel' -print -quit)"
  test -n "$XMODEL" || exit 1
  python compile_xmodel.py \
    --xmodel "$XMODEL" --arch "$ARCH" \
    --output-dir "../build/vitis_ai/compiled_zcu104/$MODEL" \
    --name "methane_$MODEL"
done
```

Saia do container e copie modelos, benchmark e dataset para a ZCU104:

```bash
scp -O -r build/vitis_ai/compiled_zcu104 root@192.168.2.100:/home/root/models/
scp -O -r Benchmark_ZCU104 root@192.168.2.100:/home/root/
scp -O -r CAMINHO_DO_STARCOP_TEST root@192.168.2.100:/home/root/STARCOP_test
```

Na placa, confirme a DPU e compile o benchmark:

```bash
xdputil query
find /home/root/models/compiled_zcu104 -name '*.xmodel'
cd /home/root/Benchmark_ZCU104
chmod +x build_zcu104.sh run_all_models.sh
./build_zcu104.sh
./benchmark_zcu104 --self-test
```

## 3. Executar o benchmark

O modo recomendado faz um sweep curto com cinco imagens, limita a dois runners para evitar falta de memória, escolhe a melhor configuração e depois processa todo o dataset com 1, 2, 3 e 4 núcleos. Energia é amostrada a cada 200 ms.

```bash
cd /home/root/Benchmark_ZCU104
./run_all_models.sh \
  --models-dir /home/root/models/compiled_zcu104 \
  --dataset /home/root/STARCOP_test \
  --out /home/root/Benchmark_ZCU104/results_all_models \
  --quick-sweep --skip-baseline --sweep-samples 5 \
  --max-runners 2 --max-concurrency 2 --pin-modes no-pin \
  --search-iterations 5 --candidate-iterations 5 \
  --warmup 2 --final-warmup 20 \
  --full-samples 0 --final-iterations 0 --final-repeats 1 \
  --power --power-sample-ms 200 --resume
```

`--skip-baseline` pula a configuração inicial do sweep; ele não pula o modelo chamado `baseline`. Para executar ou retomar apenas um modelo, use:

```bash
./run_all_models.sh \
  --model /home/root/models/compiled_zcu104/depth_reduced/methane_depth_reduced.xmodel \
  --dataset /home/root/STARCOP_test \
  --out /home/root/Benchmark_ZCU104/results_all_models \
  --quick-sweep --skip-baseline --sweep-samples 5 \
  --max-runners 2 --max-concurrency 2 --pin-modes no-pin \
  --search-iterations 5 --candidate-iterations 5 \
  --warmup 2 --final-warmup 20 \
  --full-samples 0 --final-iterations 0 --final-repeats 1 \
  --power --power-sample-ms 200 --resume
```

Para o processo continuar depois de fechar o terminal, execute-o dentro de `tmux` ou use `nohup`:

```bash
nohup ./run_all_models.sh [argumentos] > benchmark.log 2>&1 &
tail -f benchmark.log
```

O benchmark possui duas regiões de medição. `model-only` mede os dados já preparados entrando no runner e a espera pela conclusão do grafo. `end-to-end` mede leitura dos TIFFs, normalização, quantização, filas, execução e pós-processamento. Inicialização do modelo, warm-up, cálculo das métricas e gravação dos arquivos ficam fora das regiões temporizadas.

## 4. Configurações disponíveis

`run_all_models.sh` recebe `--model`, `--models-dir`, `--dataset`, `--csv` e `--out`; todas as demais opções são repassadas ao sweep.

| Opção | Significado |
|---|---|
| `--model ARQUIVO` | Executa somente um `.xmodel` |
| `--models-dir PASTA` | Encontra recursivamente todos os `.xmodel` |
| `--dataset PASTA` | Raiz dos TIFFs do dataset |
| `--csv ARQUIVO` | CSV das amostras; padrão: `<dataset>/test.csv` |
| `--out PASTA` | Diretório de resultados; use sempre o mesmo para retomar |
| `--sweep-samples N` | Quantidade de imagens distintas usadas pelo sweep |
| `--search-iterations N` | Inferências de cada configuração na busca |
| `--candidate-iterations N` | Inferências para confirmar os melhores candidatos; `0` reutiliza a busca |
| `--full-samples N` | Imagens do benchmark final; `0` usa todas |
| `--final-iterations N` | Inferências finais; `0` executa uma por imagem selecionada |
| `--warmup N` | Aquecimentos de cada teste do sweep |
| `--final-warmup N` | Aquecimentos do benchmark final; `-1` reutiliza `--warmup` |
| `--final-candidates N` | Quantidade de candidatos confirmados antes da escolha |
| `--final-repeats N` | Repetições separadas do benchmark final |
| `--max-runners N` | Máximo de runners concorrentes; `0` escolhe automaticamente |
| `--max-concurrency N` | Limite conjunto de workers e slots, de 1 a 16 |
| `--dpu-cores N` | Quantidade física de núcleos DPU; padrão da ZCU104 deste projeto: 2 |
| `--pin-modes both\|pin\|no-pin` | Testa ou fixa afinidade de CPU |
| `--quick-sweep` | Busca somente runners e 1–2 workers de pré-processamento |
| `--skip-baseline` | Remove apenas a etapa inicial de referência do sweep |
| `--baseline-repeats N` | Repetições do `model-only` na etapa baseline |
| `--baseline-e2e-passes N` | Passagens `end-to-end` na etapa baseline |
| `--power` / `--no-power` | Ativa ou desativa os sensores `hwmon` |
| `--power-sample-ms N` | Intervalo entre leituras de potência |
| `--resume` | Reutiliza execuções válidas existentes no mesmo `--out` |
| `--dry-run` | Mostra a campanha sem executar inferências |

Para uma execução manual sem sweep, use `benchmark_zcu104`. As opções exclusivas são `--profile all|baseline|max-model-only|max-e2e`, `--samples`, `--cpu-cores` ou `--threads`, `--runners`, `--pre-workers`, `--post-workers`, `--slots-per-runner`, `--iterations`, `--warmup`, `--pin`/`--no-pin` e `--validate`/`--no-validate`. O batch é sempre 1.

## 5. Resultados e diagnóstico

| Arquivo | Conteúdo |
|---|---|
| `benchmark_geral.csv` | FPS, latências, tempos dos estágios, potência e energia para 1–4 núcleos |
| `benchmark_overhead.csv` | Diferença entre `end-to-end` e `model-only` |
| `metricas_globais.csv` | TP, FP, FN, TN, precisão, recall, F1, IoU, AUPRC, FPR e acurácia |
| `metricas_por_imagem.csv` | Métricas de cada amostra |
| `all_runs.csv` | Todas as configurações testadas pelo sweep |
| `ranking_search.csv` e `ranking_final_runs.csv` | Classificação dos candidatos |
| `best_config.json` | Configuração escolhida |
| `benchmark_samples.csv` | Latência individual e tempos dos estágios |
| `benchmark_power_rails.csv` | Potência e energia por sensor da placa |
| `config.txt` | Modelo, dataset, afinidade, tensores e partições do grafo |

As métricas temporais são latência média, mínima, máxima, mediana, P95, P99, FPS e tempo total. As métricas de qualidade são TP, FP, FN, TN, precisão, recall, F1 global, F1 forte, F1 fraca, IoU, AUPRC, FPR sem pluma e acurácia. Energia inclui potência média, mínima e máxima e joules estimados durante a região medida.

Copie todos os resultados da placa para a pasta atual do computador com:

```bash
scp -O -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -r root@192.168.2.100:/home/root/Benchmark_ZCU104/results_all_models .
```

Problemas mais comuns:

| Sintoma | Ação |
|---|---|
| A placa congela com um modelo maior | Reinicie e limite para `--max-runners 2 --max-concurrency 2` |
| O `--resume` começa outro conjunto | Use exatamente o mesmo `--out` e a mesma configuração |
| `No DPU subgraph found` | Confirme `xdputil query`, o bitstream e o `arch.json` usado na compilação |
| Modelo ou dataset não encontrado | Use caminhos absolutos e confirme com `find`/`test -f` |
| Não aparecem métricas de energia | Confira `/sys/class/hwmon/hwmon*/power*_input`; a execução continua sem sensores |
| MobileNet tem maior tempo interno | Seus grafos atuais possuem operações de `upsample` executadas fora da partição DPU |

Para verificar todas as opções da versão instalada, use `./sweep_zcu104 --help`, `./benchmark_zcu104 --help` e `./run_all_models.sh --help`.

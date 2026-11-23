# CLAUDE.md — Methane_segmentation (TCC / artigo)

Segmentação de plumas de metano (STARCOP, AVIRIS, 512×512, 4 canais) com U-Nets leves,
quantizadas (Vitis AI 3.5) e medidas em GPU/CPU, ARM e DPU (ZCU104).
Idioma do projeto: **português** (docs, commits, comentários). Responda em português.

> Estado verificado em 2026-11-18. Reconfira com os comandos da seção "Checagem rápida"
> antes de confiar em qualquer caminho: este ambiente tem várias cópias do projeto e
> containers antigos apontando para diretórios errados.

## 1. Onde está cada coisa (host)

| Item | Caminho real |
|---|---|
| **Repositório (use este)** | `/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation` (disco `hdd`, 345 GB livres) |
| Cópia antiga (NÃO editar) | `/media/jacques/games/Laboratorio_LEDS/Methane-segmentation-tcc` |
| Dataset de **treino/calibração** (3425 imagens + `train.csv`, `train_easy.csv`) | `/media/jacques/games/Datasets/STARCOP_train_remaining_all` |
| Dataset de **teste** (342 imagens + `test.csv`) | `/media/jacques/games/Datasets/test/STARCOP_test` |
| venv Python (host) | `<repo>/.venv` (Python 3.10, torch 2.5.1+cu124, kornia 0.8.0, rasterio 1.4.3, smp 0.5.0, timm) |
| `<repo>/Datasets/` | **vazio de propósito** (só `.gitkeep`; ignorado pelo git) |
| `<repo>/build/` (raiz) | resíduo antigo (~92 KB). O build real é **`VitisAI/build/vitis_ai/`** |
| Disco raiz `/` | 88% cheio (13 GB livres) — não escreva coisa grande fora de `hdd`/`games` |

Cada amostra do dataset é uma pasta com `mag1c.tif`, `TOA_AVIRIS_{460,550,640}nm.tif`,
`labelbinary.tif`, `weight_mag1c.tif` (e outros). O CSV tem coluna `folder` como
`/AVIRISNG/Permian2019/<id>`; o loader (`Utils/DataLoader.py`) usa o `id`/prefixo + o
diretório raiz passado, não o `folder` literal.

Pré-processamento (idêntico em treino, host, ARM e DPU — não alterar sem refazer tudo):
canais na ordem **`mag1c, 460nm, 550nm, 640nm`**; mag1c/1750, RGB/60, clip [0, 2];
máscara = `logit > 0` seguida de abertura morfológica (kernel em cruz 3×3) nas métricas.
Teste oficial = **342 imagens** de `test.csv`, 512×512 inteiras, batch 1.

## 2. Caminhos hardcoded que dependem desta máquina

- `Testes/Teste_Unet.py:75-79` e `benchmark_nao_embarcado/benchmark_dpu.py:169` → `/media/jacques/games/Datasets/test/STARCOP_test`.
- `Treinos/Treinamento_Unet.py` usa **absoluto** (`CAMINHO_CSV` e `DIRETORIO_DADOS` em
  `/media/jacques/games/Datasets/STARCOP_train_remaining_all`), como o `main.ipynb`. Não use caminho relativo.

## 2b. Treino (`main.ipynb` → `Treinos/Treinamento_Unet.py`)

- Fluxo sincronizado com o `main.ipynb` do João (`repo_joao/`): patches 128×128, stride 64, batch 4 janelas,
  Adam lr 1e-4, validação em imagens 512 completas (`modo_treino=False`), split `GroupShuffleSplit` 15% (seed 42).
- Loss escolhida por variável do notebook, **não** na lista de modelos: `selected_loss = "BCE"` ou `"FD"`
  (FocalDice alpha 0.25, gamma 2.0). O nome do checkpoint é `<NOME>_<loss>_mag1c_rgb.pth` em `Modelos_treinados/`.
- Lista `MODELS` do notebook inclui as variantes DPU: `SegFormerB0_DPU` e `SegFormerB0_Attention_DPU`.
  `UNetResNet34_dpu` está importado mas **não** está no treino.
- Early stopping: até 200 épocas; para após 10 épocas sem melhora do F1 de validação; o checkpoint
  é salvo a cada melhora.
- Usa GPU quando disponível (`device`), com AMP apenas em CUDA (`amp_ativo = device.type == "cuda"`).
- Não usar a `Testes/` da mesma forma que o notebook do João: as células de teste do João passam `device`
  e `Testes/Teste_Unet.py::avaliar_e_visualizar` ainda não aceita esse argumento.
- `VitisAI/*.py` (container) esperam `/dataset_STARCOP/train.csv` e `/dataset_STARCOP`;
  o teste deve estar em `/dataset_test`.
- `.vscode/settings.json` tem `cmake.sourceDirectory` de outra máquina (`/home/thiago/...`) — ignorar.
- `GUIA_PROJETO_ZCU104.md` cita `/home/root/thiago/...` — são caminhos **na placa**, não no host.

## 3. Python / host

```bash
cd /media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation
source .venv/bin/activate        # (README antigos dizem "venv/"; o nome real é .venv)
```
- Rode benchmarks como módulo, da raiz: `python -m benchmark_nao_embarcado.benchmark_geral ...`.
  Ao copiar comandos dos READMEs, troque `venv/bin/python` por `.venv/bin/python`.
- O `.venv` **não** tem `onnx`, `onnxruntime` nem `pynvml` instalados. Instale antes de exportar
  ONNX (`benchmark_arm/exportar_to_onnx/scripts/exportar.py`, versões usadas: onnx 1.16.2,
  onnxruntime 1.20.1, opset 16/IR 8) ou de medir energia NVML.
- `requirements.txt` está congelado (~2,9 KB, versões do artigo); não é garantia de que o
  `.venv` atual o reproduza exatamente.

### GPU — RESOLVIDO em 2026-11-18
GPU NVIDIA RTX 4070 SUPER, driver 580.142, `nvidia-smi` funciona. Kernel 7.0.0-43.
A falha anterior era Secure Boot: o módulo NVIDIA não era aceito porque a chave de assinatura
(`/var/lib/shim-signed/mok/MOK.der`) não estava registrada. Carregamento manual pelo usuário:
`sudo modprobe nvidia_uvm && sudo modprobe nvidia`. **Após trocar kernel ou driver**, o módulo pode
ser rejeitado de novo; o fix é registrar a chave MOK e reiniciar, ou desativar Secure Boot.
Não faça `apt`/`modprobe`/reboot sem o usuário pedir.
Treino e benchmark CUDA dependem disso; o código já é consciente de dispositivo (AMP só em CUDA).

### Energia da CPU (RAPL)
`/sys/class/powercap/intel-rapl:0/energy_uj` é `root:root 0400`; **reseta a cada boot**.
Antes de medir energia (precisa do usuário/sudo):
```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
```
Energia CPU+GPU = RAPL (pacote CPU) + NVML (GPU); **não** é consumo na tomada. A máquina tem
12 CPUs e 62 GB RAM; os benchmarks usam `--num-threads 4`.

## 4. Vitis AI (Docker)

Imagens: `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306` (usar esta), mais
`xiinx/vitis-ai-gpu-pytorch-base` (nome com typo, quebrada/ não usar) e `nvidia/cuda:11.7.1`.

Container "principal": **`methane-vitis-ai-cpu`** (recriado em 2026-11-18, root, mounts corretos):
- `/workspace` ← `/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation` (RW)
- `/dataset_STARCOP` ← `/media/jacques/games/Datasets/STARCOP_train_remaining_all` (RO, 3428 itens)
- `/dataset_test` ← `/media/jacques/games/Datasets/test/STARCOP_test` (RO, 343 itens)

O container antigo foi renomeado para **`methane-vitis-ai-cpu-old`** (parado, mounts errados) e
pode ser removido quando o usuário quiser (`docker rm methane-vitis-ai-cpu-old`). Os containers
`methane-vitis-artigo-*` e `methane-vitis-ai-dpu*` (parados) apontam para a cópia antiga em
`games/Laboratorio_LEDS/...` em `/workspace`. Não os reutilize.

Para recriar, se preciso (mounts só são definidos na criação):

```bash
docker run -dit --name methane-vitis-ai-cpu \
  -v /media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation:/workspace \
  -v /media/jacques/games/Datasets/STARCOP_train_remaining_all:/dataset_STARCOP:ro \
  -v /media/jacques/games/Datasets/test/STARCOP_test:/dataset_test:ro \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306 bash
```

Dentro do container:
```bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch
cd /workspace/VitisAI
test -f /dataset_STARCOP/train.csv && test -f /dataset_test/test.csv && echo OK
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json   # existe; target DPUCZDX8G_ISA1_B4096
```
- Ambiente `vitis-ai-pytorch` (verificado): torch 1.13.1+cpu, kornia 0.6.12, rasterio 1.3.11, pytorch-nndct 3.5.0.
- **kornia e rasterio NÃO vêm na imagem**: foram instalados com `pip install kornia==0.6.12 rasterio==1.3.11`
  dentro do env. Se o container for recriado, instale de novo (o `docker run` acima não os traz).
- O container não tem GPU; toda quantização/avaliação INT8 é **simulada em CPU**.
- Perfil de calibração do artigo (validado pelo CLI, outros valores são rejeitados): 100 imagens
  de **treino**, seed 12345, política `layer_mse`, 512 ativações/camada/imagem, sem refinamento,
  **nunca amostras do test**. Diretórios já calibrados não são sobrescritos (use dir novo).
- Seleção atual do artigo (5 modelos): `attentiongates_dpu_bce_artigo`,
  `attentiongates_dpu_focaldice_artigo` (remaining_all), `mobilenet_v3_dpu_bce_artigo`,
  `mobilenet_v3_dpu_focaldice_artigo`, `hyperstarcop_artigo_512`.
  Saídas: `VitisAI/build/vitis_ai/{quantize_artigo,evaluation_artigo,compiled_artigo,logs_artigo,calibration_artigo}`.
  Pastas sem sufixo (`quantize/`, `compiled/`, `evaluation/`, `inspect/`) contêm os resultados de inspeção, quantização, compilação e avaliação dos modelos da DPU que não são do artigo (baseline, depth_reduced, skip_connections, resnet34_dpu, MobileNetV3 e AttentionGates de experimentos). Cada etapa tem `logs/` quando há logs.
- `iniciar_vitisAI_desktop.txt` ainda usa os 5 modelos antigos e `--output-dir ../build/vitis_ai/quantize`
  (caminho errado vs. `VitisAI/build/...`); prefira `VitisAI/README.md`.
- Incompatibilidades DPU (ver `Modelos/INCOMPATIBILIDADES_VITIS_AI.txt` e `UNet_MobileNetV3_AttentionGates_DPU.md`):
  `Upsample` bilinear com `align_corners=True`, e Sigmoid/Mul com broadcast de 1 canal vão para a
  CPU; por isso existem as variantes `*_DPU` (Hardsigmoid, mapa de atenção por canal,
  `align_corners=False`). Para a DPU use sempre as classes `UNetMobileNetV3_dpu` e
  `UNetMobileNetV3AttentionGatesDPU`.

## 5. Git / LFS

- Branch `main`, árvore limpa. `*.pth`, `*.xmodel`, `*.ckpt` são **Git LFS** (`git-lfs 3.4.1`
  instalado; `.git` ~106 MB, 55 objetos LFS). Confirme que os arquivos não são ponteiros
  (`head -c 100 arquivo.pth` não deve mostrar `version https://git-lfs`); se forem, `git lfs pull`.
- Gitignored: `.venv/`, `venv/`, `Datasets/*`, `STARCOP_test/`, `STARCOP_mini/`, `zcu104_sysroot/`,
  ONNX convertidos (`modelos_convertidos_onnx/*.onnx`), binário/resultados ARM locais e `__pycache__`.
- Commits e PRs: mensagens curtas em inglês/português como no histórico; não commitar sem pedido.
- `README_COMMITS.md` parou em 05/10; os commits seguintes não estão listados.

## 6. Placa ZCU104 e cross-compile

- IP usado nos docs: `192.168.2.100` (usuário `root`). **Não está respondendo agora** (ping 100% perdido)
  — placa desligada ou rede diferente. Há `~/.ssh/id_ed25519`. Use `scp -O` (obrigatório na placa).
- Caminhos na placa: fontes `/home/root/thiago/benchmark/codigos_c/`, modelos
  `/home/root/thiago/benchmark/modelos/`, teste `/home/root/thiago/STARCOP_test/`.
- **`zcu104_sysroot/` e cross-compilador `aarch64-linux-gnu-g++` não existem** neste host. Os C++
  (`benchmark_zcu104/codigos_c`, `benchmark_arm/codigos_c`) são compilados **na própria placa**
  (`build_zcu104.sh`, `build.sh`) ou exigem instalar sysroot/toolchain antes.
- ARM mede ONNX Runtime **FP32** no Cortex-A53; DPU mede XModel **INT8**; host mede PyTorch **FP32**.
  Protocolos diferentes — nunca compare FPS/energia entre plataformas sem citar modo
  (`model_only` vs `end_to_end`) e domínio do sensor.

## 7. Checkpoints (`Modelos_treinados/`)

Os do artigo estão duplicados em `Modelo_artigo/` e na raiz; **o alias
`attentiongates_dpu_bce_artigo` agora resolve para `UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth`**
(retreinado). A coleta CUDA/CPU antiga desse alias usou o checkpoint anterior e está
desatualizada; o benchmark CPU/GPU do novo checkpoint está **pendente** (e depende do driver GPU).
Há checkpoints com nomes quase iguais (`..._old.pth`, `MobileNetV3_AttentionGates_*` vs
`UNetMobileNetV3AttentionGates_*`, `UnetMobilenetV3...` com `U` minúsculo). Sempre confirme o
arquivo exato e seu hash; vários READMEs citam `*_easy_remaining*` que **não existem mais** no repo.
HyperSTARCOP: `Modelos_treinados/HyperSTARCOP_oficial/{config.yaml,final_checkpoint_model.ckpt}`.

## 8. Checagem rápida (rode antes de começar)

```bash
cd /media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation
wc -l /media/jacques/games/Datasets/test/STARCOP_test/test.csv        # 343 (342+cabeçalho)
wc -l /media/jacques/games/Datasets/STARCOP_train_remaining_all/train.csv   # 3426
.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
nvidia-smi | head -3
ls -l /sys/class/powercap/intel-rapl:0/energy_uj                       # precisa -r--r--r--
docker ps --format '{{.Names}}'; docker exec <ctr> ls /dataset_STARCOP | head -2
docker exec <ctr> conda run -n vitis-ai-pytorch python -c "import kornia"
ping -c1 -W1 192.168.2.100
df -h / /media/jacques/hdd | tail -2
```

## 9. Armadilhas resumidas

1. Container `methane-vitis-ai-cpu` tem `/dataset_STARCOP` vazio e sem `/dataset_test` → recrie com os mounts da seção 4.
2. Sem `kornia` no env conda do container → `pip install kornia==0.6.12`.
3. Driver NVIDIA sai do ar após troca de kernel (Secure Boot) → checar `nvidia-smi` antes de `--device cuda` ou treino.
4. RAPL só legível após `sudo chmod` (reseta no boot).
5. Docs falam em `venv/` e em `build/vitis_ai/...` na raiz; o real é `.venv/` e `VitisAI/build/vitis_ai/...`.
6. `Treinamento_Unet.py` usa caminho absoluto do dataset de treino; não recolocar caminho relativo.
7. Não use a cópia em `games/Laboratorio_LEDS/...` nem containers antigos (workspace desatualizado).
8. Nunca coloque amostras do test na calibração; nunca sobrescreva diretórios já calibrados/medidos.
9. Disco `/` quase cheio: logs, ONNX e builds vão para `hdd`.

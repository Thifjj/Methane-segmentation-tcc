# Conversao para Vitis AI 3.5 / ZCU104

## Notebook

Na primeira execucao, crie o container com o Vitis AI, este projeto e o dataset
completo montados:

```bash
docker run -it \
  --name methane-vitis-ai-cpu \
  --network host \
  -v /dev/shm:/dev/shm \
  -v "/home/thiago/Documents/aplicativos/Vitis-AI":/vitis_ai_home \
  -v "/home/thiago/Documents/Laboratorio_LEDS/Projetos_aceleradores/Segmentacao_de_metano/Joao/projeto/Methane-segmentation-tcc":/workspace \
  -v "/home/thiago/Documents/STARCOP_DATASET":/dataset_STARCOP:ro \
  -w /workspace/VitisAI \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306 \
  bash
```

Nas execucoes seguintes:

```bash
docker start methane-vitis-ai-cpu
docker exec -it -w /workspace/VitisAI methane-vitis-ai-cpu bash
conda activate vitis-ai-pytorch
```

Dentro do container, o projeto fica em `/workspace`, o dataset em
`/dataset_STARCOP` e o codigo-fonte do Vitis AI em `/vitis_ai_home`.

Os scripts sao a interface principal e devem ser executados **dentro do container PyTorch do Vitis AI**. Troque somente `--model` entre `baseline`, `depth_reduced`, `skip_connections`, `mobilenet_v2` e `mobilenet_v3`. Por padrao, cada nome seleciona seu checkpoint em `Modelos_treinados/` e usa os quatro canais `mag1c,R,G,B` do projeto.

> Use uma versao do container Vitis AI compativel com a imagem/bitstream instalado na ZCU104. O `arch.json` tambem deve ser exatamente o dessa DPU; nao use um arquivo apenas porque ele tem o nome ZCU104.

Este fluxo usa o container oficial **CPU** `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306`. A GPU NVIDIA usada no treinamento nao e necessaria para quantizar ou compilar. A mensagem `No CUDA runtime is found` e esperada neste container.

## 0. Entrar no ambiente Vitis AI

O container CPU deste projeto chama-se `methane-vitis-ai-cpu`. No terminal do
host, inicie-o e entre nele:

```bash
docker start methane-vitis-ai-cpu
docker exec -it methane-vitis-ai-cpu bash
```

Os comandos restantes deste guia devem ser executados **dentro do container**.
Ative o ambiente e entre na pasta dos scripts:

```bash
conda activate vitis-ai-pytorch
cd /workspace/VitisAI
```

O prompt deve ficar parecido com:

```text
(vitis-ai-pytorch) vitis-ai-user@jacquespc:/workspace/VitisAI$
```

Confirme os arquivos antes de começar:

```bash
test -f /dataset_STARCOP/train.csv && echo "CSV OK"
test -f ../Modelos_treinados/UNET_depth_reduced_mag1c_rgb.pth && echo "CHECKPOINT OK"
python -c 'import torch, rasterio; print("PyTorch", torch.__version__); print("Rasterio", rasterio.__version__)'
```

## 1. Inspecionar os modelos

Descubra os targets presentes no container e use o target correspondente a sua DPU. Exemplo (o nome exato depende da versao/imagem):

```bash
for MODEL in baseline depth_reduced skip_connections mobilenet_v2 mobilenet_v3
do
  python inspect_model.py \
    --model "$MODEL" \
    --target DPUCZDX8G_ISA1_B4096 \
    --output-dir build/vitis_ai/inspect
done
```

Consulte `build/vitis_ai/inspect/<modelo>/` e as imagens geradas.
`ConvTranspose2d`, interpolacao bilinear e operadores de concatenacao podem
produzir subgrafos fora da DPU; o Inspector e a fonte de verdade.

## 2. Calibrar INT8

Use imagens reais e representativas. O preprocessamento e exatamente o `DataNormalizer` usado no teste original.
O subconjunto e sorteado de forma reproduzivel (`--seed 12345`) e os rotulos
nao sao carregados, pois nao participam da calibracao. Por padrao, dois workers
antecipam a leitura dos GeoTIFFs; use `--num-workers 0` se o ambiente limitar
multiprocessamento. `--batch-size` permanece em 1 por seguranca de memoria, mas
pode ser aumentado durante a calibracao apos medir o consumo de RAM.

Calibre e, em seguida, exporte cada modelo. Se qualquer etapa falhar, `set -e`
interrompe o loop e evita exportar com uma calibracao ausente ou incompleta.

```bash
set -e

for MODEL in mobilenet_v2 mobilenet_v3
do
  echo "=== CALIBRANDO $MODEL ==="
  python3 quantize_model.py \
    --model "$MODEL" \
    --quant-mode calib \
    --csv /dataset_STARCOP/train.csv \
    --data-root /dataset_STARCOP \
    --subset-len 1000 \
    --target DPUCZDX8G_ISA1_B4096 \
    --output-dir build/vitis_ai/quantize

  echo "=== EXPORTANDO $MODEL ==="

  python quantize_model.py \
    --model "$MODEL" \
    --quant-mode test \
    --csv /dataset_STARCOP/train.csv \
    --data-root /dataset_STARCOP \
    --target DPUCZDX8G_ISA1_B4096 \
    --output-dir build/vitis_ai/quantize \
    --deploy
done
```

O deploy força automaticamente `batch_size=1` e uma unica inferencia. Confira
os arquivos INT8 exportados:

```bash
find build/vitis_ai/quantize -name '*_int.xmodel'
```

## 3. Compilar para o bitstream da ZCU104

O arquivo exportado pelo quantizador ainda nao e o artefato final da placa. Ele
precisa ser compilado com o `arch.json` **exato do bitstream/DPU instalado na
ZCU104**. O nome do target usado acima nao substitui essa verificacao.

Procure as configuracoes disponiveis:

```bash
find /opt/vitis_ai -path '*ZCU104*' -name arch.json 2>/dev/null
```

Compile os cinco modelos diretamente para a ZCU104. Execute dentro de
`/workspace/VitisAI`; tanto a entrada quanto a saida ficam no `build` dessa
pasta:

```bash
set -e
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
test -f "$ARCH" || { echo "arch.json da ZCU104 ausente"; exit 1; }

for MODEL in baseline depth_reduced skip_connections mobilenet_v2 mobilenet_v3
do
  XMODEL="$(find "build/vitis_ai/quantize/$MODEL" -maxdepth 1 -name '*_int.xmodel' -print -quit)"
  test -n "$XMODEL" || { echo "XModel ausente para $MODEL"; exit 1; }

  python compile_xmodel.py \
    --xmodel "$XMODEL" \
    --arch "$ARCH" \
    --output-dir "build/vitis_ai/compiled_zcu104/$MODEL" \
    --name "methane_$MODEL"
done
```

Confira os artefatos compilados:

```bash
find build/vitis_ai/compiled_zcu104 -name '*.xmodel'
```

Uma compilacao cujo arquivo de saida esteja atualizado e ignorada
automaticamente. Acrescente `--force` ao comando para recompilar.

## 4. Copiar para a placa

Saia do container:

```bash
exit
```

No host, copie os modelos, substituindo o usuario e o IP da ZCU104:

```bash
scp -r VitisAI/build/vitis_ai/compiled_zcu104 \
  root@IP_DA_ZCU104:/home/root/models/
```

Copiar o modelo nao inicia a inferencia. A placa tambem precisa do Vitis AI
Runtime compativel com o bitstream e de um aplicativo VART que carregue os
quatro canais, aplique o mesmo `DataNormalizer`, execute a DPU e gere a mascara.

Na placa, primeiro confirme que a DPU esta acessivel e que os modelos chegaram:

```bash
xdputil query
find /home/root/models/compiled_zcu104 -name '*.xmodel'
xdputil xmodel /home/root/models/compiled_zcu104/depth_reduced/methane_depth_reduced.xmodel -l
```

Para medir apenas o desempenho da DPU com entrada sintetica, consulte a sintaxe
instalada e execute o benchmark com uma thread:

```bash
xdputil benchmark --help
xdputil benchmark /home/root/models/compiled_zcu104/depth_reduced/methane_depth_reduced.xmodel 1
```

Esse benchmark nao mede leitura dos GeoTIFFs, normalizacao, pos-processamento,
IoU ou F1. Para o benchmark completo, copie `benchmark_zcu104/codigos_c/` para
`/home/root/thiago/benchmark/` e siga
[`benchmark_zcu104/codigos_c/README.md`](../benchmark_zcu104/codigos_c/README.md).
O código atual mede os datasets `STARCOP_test` e `dataset_starcop` e mantém
saídas separadas para cada dataset.

## 5. Como adicionar manualmente um modelo ao código

Os scripts `inspect_model.py` e `quantize_model.py` usam o dicionário `MODEL_REGISTRY`, localizado em `VitisAI/common.py`. Para disponibilizar uma arquitetura nova por meio de `--model`, é necessário cadastrar sua classe e seu checkpoint nesse dicionário.

### Passo 1 — adicionar a implementação

Coloque o arquivo Python da arquitetura em `Modelos/`. A classe precisa herdar de `torch.nn.Module`, implementar `forward()` e aceitar os argumentos utilizados pelo carregador:

```python
class MeuModelo(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        # Definição das camadas

    def forward(self, x):
        # Inferência
        return logits
```

Para os dados `mag1c + RGB` deste projeto, o modelo recebe quatro canais e produz um canal de logits de segmentação.

### Passo 2 — adicionar o checkpoint

Copie o `state_dict` treinado para `Modelos_treinados/`:

```text
Modelos_treinados/MeuModelo_mag1c_rgb.pth
```

O checkpoint deve pertencer exatamente à arquitetura adicionada. Alterar somente o nome de um `.pth` não converte seus pesos para outra arquitetura.

### Passo 3 — importar a classe

Abra `VitisAI/common.py` e adicione o import junto aos imports dos outros modelos:

```python
from Modelos.MeuModelo import MeuModelo
```

A estrutura do import é:

```python
from Modelos.NOME_DO_ARQUIVO import NOME_DA_CLASSE
```

### Passo 4 — registrar classe e checkpoint

No mesmo arquivo, encontre `MODEL_REGISTRY` e acrescente uma entrada:

```python
MODEL_REGISTRY = {
    "baseline": (UNetBaseline, "UNET_mag1c_rgb.pth"),
    "depth_reduced": (UNetDepthReduced, "UNET_depth_reduced_mag1c_rgb.pth"),
    "skip_connections": (UNetElementWise, "UNET_SkipConnections_mag1c_rgb.pth"),
    "mobilenet_v2": (UNetMobileNetV2, "Mobile_Net_v2_mag1c_rgb.pth"),
    "mobilenet_v3": (UNetMobileNetV3, "Mobile_Net_v3_mag1c_rgb.pth"),
    "meu_modelo": (MeuModelo, "MeuModelo_mag1c_rgb.pth"),
}
```

Cada registro segue o formato:

```python
"nome_usado_no_argumento": (ClasseDoModelo, "checkpoint.pth")
```

Portanto, `"meu_modelo"` será o valor fornecido a `--model`. O nome deve ser único e o checkpoint é procurado automaticamente dentro de `Modelos_treinados/`.

O baseline deste projeto foi cadastrado dessa forma:

```python
from Modelos.UNet_baseline import UNetBaseline

MODEL_REGISTRY = {
    "baseline": (UNetBaseline, "UNET_mag1c_rgb.pth"),
    # Outros modelos...
}
```

### Passo 5 — verificar o cadastro

Dentro do container Vitis AI e da pasta `/workspace/VitisAI`, execute:

```bash
python inspect_model.py --help
```

O novo nome deverá aparecer nas opções de `--model`. Em seguida, teste o carregamento e a compatibilidade com a DPU:

```bash
python inspect_model.py \
  --model meu_modelo \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/inspect
```

Se aparecer erro em `load_state_dict`, a classe não corresponde ao checkpoint, o número de canais está incorreto ou os pesos foram salvos usando uma estrutura diferente.

### Passo 6 — calibrar e exportar

Depois que a inspeção funcionar, calibre o modelo:

```bash
python quantize_model.py \
  --model meu_modelo \
  --quant-mode calib \
  --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP \
  --subset-len 1000 \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/quantize
```

Finalmente, teste e exporte o `.xmodel` INT8 usando exatamente o mesmo nome e target:

```bash
python quantize_model.py \
  --model meu_modelo \
  --quant-mode test \
  --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/quantize \
  --deploy
```

Não é necessário modificar `inspect_model.py` nem `quantize_model.py`: ambos obtêm automaticamente os modelos disponíveis a partir de `MODEL_REGISTRY`.

## 6. Diagnostico rapido

- `Nenhuma amostra valida`: confirme o CSV, o volume `/dataset_STARCOP` e os quatro TIFFs exigidos em uma pasta de amostra.
- `FileNotFoundError: /dataset_STARCOP/train.csv`: o container foi criado sem o volume correto; recrie-o conforme [`iniciar_vitisAI_desktop.txt`](../iniciar_vitisAI_desktop.txt).
- `FutureWarning` vindo de `scipy.stats.mode`: e um aviso de compatibilidade da versao interna do quantizador; aguarde a linha de progresso antes de concluir que travou.
- Leitura lenta ou erro de multiprocessing: repita com `--num-workers 0`.
- `No CUDA runtime is found`: normal no container CPU usado neste projeto.
- Nao execute `test --deploy` sem a calibracao correspondente no mesmo `--output-dir`.

Os diretorios `build/vitis_ai/quantize`, `build/vitis_ai/inspect` e
`build/vitis_ai/compiled_zcu104` sao artefatos gerados. Preserve-os para o
deploy, mas evite adiciona-los ao Git sem uma decisao explicita sobre o tamanho
do repositorio.

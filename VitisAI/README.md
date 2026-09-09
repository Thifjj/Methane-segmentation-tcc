# Conversao para Vitis AI / ZCU104

Os scripts sao a interface principal e devem ser executados **dentro do container PyTorch do Vitis AI**. Troque somente `--model` entre `baseline`, `depth_reduced`, `skip_connections`, `mobilenet_v2` e `mobilenet_v3`. Por padrao, cada nome seleciona seu checkpoint em `Modelos_treinados/` e usa os quatro canais `mag1c,R,G,B` do projeto.

> Use uma versao do container Vitis AI compativel com a imagem/bitstream instalado na ZCU104. O `arch.json` tambem deve ser exatamente o dessa DPU; nao use um arquivo apenas porque ele tem o nome ZCU104.

## Abrindo VITIS AI no seu workspace
```bash
  docker run -it \
    --name methane-vitis-ai \
    --network host \
    --ipc host \
    -v "/media/thifj/nvme2/vitisai_3_5/Vitis-AI:/vitis_ai_home" \
    -v "$PWD:/workspace" \
    -w /workspace \
    xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306 \
    bash
```

## rodando apos criar
```bash
docker start methane-vitis-ai
docker exec -it methane-vitis-ai bash
conda activate vitis-ai-pytorch
``` 

## 1. Inspecionar

Descubra os targets presentes no container e use o target correspondente a sua DPU. Exemplo (o nome exato depende da versao/imagem):

```bash
python VitisAI/inspect_model.py --model depth_reduced --target DPUCZDX8G_ISA1_B4096
```

```bash
python VitisAI/inspect_model.py --model skip_connections --target DPUCZDX8G_ISA1_B4096
```

```bash
python VitisAI/inspect_model.py --model mobilenet_v2 --target DPUCZDX8G_ISA1_B4096
```

```bash
python VitisAI/inspect_model.py --model mobilenet_v3 --target DPUCZDX8G_ISA1_B4096
```

```bash
python VitisAI/inspect_model.py --model baseline --target DPUCZDX8G_ISA1_B4096
```

```bash
Modelos Disponiveis:
  depth_reduced
  skip_connections
  mobilenet_v2
  mobilenet_v3
  baseline
```

Consulte `build/vitis_ai/inspect/<modelo>/inspect_results.txt` e as imagens geradas. `ConvTranspose2d`, interpolacao bilinear e operadores de concatenacao podem produzir subgrafos fora da DPU; o Inspector e a fonte de verdade para o target escolhido.

## 2. Calibrar INT8

Use imagens reais e representativas. O preprocessamento e exatamente o `DataNormalizer` usado no teste original.

```bash
python VitisAI/quantize_model.py \
  --model depth_reduced \
  --quant-mode calib \
  --csv STARCOP_mini/train_mini10.csv \
  --data-root STARCOP_mini \
  --subset-len 100 \
  --target DPUCZDX8G_ISA1_B4096
```

## 3. Testar e exportar o xmodel INT8

Execute com o mesmo modelo, canais, target e diretorio de saida usados na calibracao:

```bash
python VitisAI/quantize_model.py \
  --model depth_reduced \
  --quant-mode test \
  --csv STARCOP_mini/test_mini10.csv \
  --data-root STARCOP_mini \
  --subset-len 1 \
  --target DPUCZDX8G_ISA1_B4096 \
  --deploy
```

## 4. Compilar para o bitstream da ZCU104

Localize primeiro o `arch.json` correto no container ou extraia-o do hardware exportado:

```bash
find /opt/vitis_ai -path '*ZCU104*' -name arch.json 2>/dev/null
python VitisAI/compile_xmodel.py \
  --xmodel build/vitis_ai/quantize/depth_reduced/UNetDepthReduced_int.xmodel \
  --arch /caminho/exato/para/arch.json \
  --name methane_depth_reduced
```

O nome do xmodel exportado pode variar conforme a versao do Vitis AI; confirme com `find build/vitis_ai -name '*_int.xmodel'`.

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

Dentro do container Vitis AI, execute:

```bash
python VitisAI/inspect_model.py --help
```

O novo nome deverá aparecer nas opções de `--model`. Em seguida, teste o carregamento e a compatibilidade com a DPU:

```bash
python VitisAI/inspect_model.py \
  --model meu_modelo \
  --target DPUCZDX8G_ISA1_B4096
```

Se aparecer erro em `load_state_dict`, a classe não corresponde ao checkpoint, o número de canais está incorreto ou os pesos foram salvos usando uma estrutura diferente.

### Passo 6 — calibrar e exportar

Depois que a inspeção funcionar, calibre o modelo:

```bash
python VitisAI/quantize_model.py \
  --model meu_modelo \
  --quant-mode calib \
  --csv STARCOP_mini/train_mini10.csv \
  --data-root STARCOP_mini \
  --subset-len 100 \
  --target DPUCZDX8G_ISA1_B4096
```

Finalmente, teste e exporte o `.xmodel` INT8 usando exatamente o mesmo nome e target:

```bash
python VitisAI/quantize_model.py \
  --model meu_modelo \
  --quant-mode test \
  --csv STARCOP_mini/test_mini10.csv \
  --data-root STARCOP_mini \
  --subset-len 1 \
  --target DPUCZDX8G_ISA1_B4096 \
  --deploy
```

Não é necessário modificar `inspect_model.py` nem `quantize_model.py`: ambos obtêm automaticamente os modelos disponíveis a partir de `MODEL_REGISTRY`.

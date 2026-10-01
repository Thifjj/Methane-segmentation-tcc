# Treinamento alinhado ao HyperSTARCOP

Referência: repositório oficial spaceml-org/STARCOP, commit c4789268a3fa0395f92357429052f6f5fc748acb. Scripts: `bash/bash_train_example.sh`, `scripts/train.py`, `scripts/configs/config.yaml`, `starcop/data/datamodule.py` e `starcop/models/model_module.py`.

## Padrões do protocolo

- Treino: todos os registros do CSV, patches 128×128 com stride 64, um patch por item do DataLoader.
- Batch de 32 significa 32 patches, sem multiplicação por 49.
- Presença de pluma: fração positiva > 10/64²; para 128×128, mais de 40 pixels. WeightedRandomSampler com reposição e pesos inversos da frequência de cada classe: aproximadamente 50% de cada grupo em expectativa, não necessariamente em cada batch.
- Loss BCE: BCEWithLogitsLoss(reduction="none", pos_weight=1), multiplicada por weight_mag1c e depois média. Focal Dice continua disponível com seus parâmetros anteriores.
- Augmentações: rotação ±90° com p=0,5, flip horizontal e vertical com p=0,5. O mapa de pesos é tratado como input e interpolado como no oficial; o gabarito é mask.
- Adam com LR inicial 1e-4. ReduceLROnPlateau pela val_loss, fator 0,5, paciência 4, atualizado ao final da época.
- Checkpoint pela menor val_loss. F1 global acumulado é informativo, sem abertura morfológica na validação. A função antiga calcular_f1_score permanece disponível, mas não seleciona checkpoints.
- Validação em imagens completas 512×512 do test.csv, a cada meia época e ao final; até 15 épocas.
- Precisão FP32 por padrão. AMP, CPU/CUDA, batch, workers, loss, pos_weight, epochs e early stopping são configuráveis.

## Particularidades do código oficial

O script cria EarlyStopping(patience=8), mas não o inclui nos callbacks do Trainer. Por isso early_stopping=False reproduz o script publicado; True permite usar essa opção por verificações de validação.

O protocolo oficial usa test.csv para validar e selecionar pesos. Nesse modo, esse conjunto também participa da seleção do modelo e não é uma avaliação independente. validation_mode="split" preserva a alternativa anterior de separar 15% das imagens por folder, seed 42; a validação segue em imagens completas.

## Opções existentes preservadas

A arquitetura continua sendo o modelo selecionado no notebook. MobileNetV3, attention gates e demais modelos não são convertidos para a MobileNetV2 do HyperSTARCOP. A normalização e a ordem dos canais fornecida pelo usuário são mantidas; o oficial mag1c+RGB usa mag1c, 640nm, 550nm, 460nm, enquanto o notebook usa mag1c, 460nm, 550nm, 640nm de maneira consistente no treino e teste.

starting_point continua aceito por compatibilidade, mas seu antigo F1 não é usado como limite de val_loss. resume=True carrega pesos e mede sua val_loss inicial; não restaura otimizador nem scheduler. Os checkpoints continuam sendo state_dicts compatíveis com os testes existentes. Um arquivo .training.json registra configurações e histórico.

No main.ipynb, CONFIG_TREINO e LOSS_TREINO controlam os treinos existentes. Os nomes incluem _hyperstarcop_bce ou _hyperstarcop_focal_dice para preservar os checkpoints anteriores. Para alterar a loss, execute novamente a célula de configuração. Esta alteração não inicia treinamento automaticamente.


## Opções de desempenho locais

O carregamento agora aceita prefetch_factor=4, persistent_workers=True, cache_max_gb=2.0 e val_cache_max_gb=0.25. O notebook usa os oito workers escolhidos pelo usuário. Os caches são LRU em RAM, limitados por worker; com oito leitores de treino e oito de validação, totalizam no máximo 18 GiB de arrays em cache. São otimizações locais da execução, não parâmetros estatísticos do artigo. Os patches, amostragem, loss, AMP e tqdm são preservados. Veja o detalhamento e as limitações no README.md.

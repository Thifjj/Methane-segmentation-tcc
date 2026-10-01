# Treinamento dos modelos de segmentação

`Treinamento_Unet.py` é executado pelo `main.ipynb`, no Python da `.venv`, usando a GPU NVIDIA do host quando selecionada. Esse ambiente é separado do container Vitis AI usado para quantização e compilação.

## Origem das mudanças: HyperSTARCOP

O protocolo foi adaptado diretamente das regras, fórmulas e configurações do **código oficial do HyperSTARCOP**, publicado com o artigo **Semantic segmentation of methane plumes with hyperspectral machine learning models**, Růžička et al., Scientific Reports, 2023. A implementação local continua em PyTorch e mantém a escolha das arquiteturas existentes; não é uma cópia integral do treinamento Lightning nem uma troca de todos os modelos pelo HyperSTARCOP.

Referência verificada: commit `c4789268a3fa0395f92357429052f6f5fc748acb` de [spaceml-org/STARCOP](https://github.com/spaceml-org/STARCOP/tree/c4789268a3fa0395f92357429052f6f5fc748acb).

| Arquivo oficial | O que foi usado |
|---|---|
| [bash/bash_train_example.sh](https://github.com/spaceml-org/STARCOP/blob/c4789268a3fa0395f92357429052f6f5fc748acb/bash/bash_train_example.sh) | Experimento HyperSTARCOP mag1c+RGB: `pos_weight=1`, pesos por pixel, amostragem balanceada, 15 épocas e validação a cada meia época. |
| [scripts/configs/config.yaml](https://github.com/spaceml-org/STARCOP/blob/c4789268a3fa0395f92357429052f6f5fc748acb/scripts/configs/config.yaml) | Batch 32, workers 4, patches 128, sobreposição 64, Adam e scheduler. O comando HyperSTARCOP sobrescreve o `pos_weight=15` genérico do YAML para 1. |
| [starcop/data/datamodule.py](https://github.com/spaceml-org/STARCOP/blob/c4789268a3fa0395f92357429052f6f5fc748acb/starcop/data/datamodule.py) | Um item por patch, definição de `has_plume`, WeightedRandomSampler, augmentações e validação por imagens completas. |
| [starcop/models/model_module.py](https://github.com/spaceml-org/STARCOP/blob/c4789268a3fa0395f92357429052f6f5fc748acb/starcop/models/model_module.py) | BCE ponderada, matriz de confusão acumulada e ReduceLROnPlateau. |
| [scripts/train.py](https://github.com/spaceml-org/STARCOP/blob/c4789268a3fa0395f92357429052f6f5fc748acb/scripts/train.py) | Checkpoint pela menor `val_loss`, intervalo de validação e comportamento real do early stopping. |

Artigo: [Scientific Reports, DOI 10.1038/s41598-023-44918-6](https://doi.org/10.1038/s41598-023-44918-6).

## Todas as mudanças e seus motivos

A coluna “antes” representa a versão imediatamente anterior à adoção desse protocolo: batch de 4 imagens, workers 4. Os dois experimentos anteriores executados por este projeto usaram batch de 16 imagens e workers 6; isso é um histórico diferente dos padrões do código imediatamente anterior.

| Item | Antes | Agora | Motivo |
|---|---|---|---|
| Conjunto de treino padrão | 2.911 imagens após reservar 15%. | Todas as 3.425 imagens do CSV. | Reproduzir o uso do train.csv inteiro pelo oficial. |
| Conjunto de validação padrão | 514 imagens reservadas do treino. | 342 imagens do test.csv externo. | Reproduzir a validação usada no código oficial. O modo split mantém a alternativa anterior. |
| Unidade de amostra | Uma imagem retornava seus 49 patches juntos. | Uma linha por patch, lido individualmente pelo dataset. | Permitir que a amostragem balanceada opere sobre cada patch. |
| Tamanho/stride | 128×128 / 64. | Mantidos. | Já correspondiam à configuração oficial. |
| Batch de treino | 4 imagens, equivalentes a 196 patches por atualização. | 32 patches independentes. | Igualar a unidade e o tamanho de batch oficial. Nos experimentos antigos com batch 16 eram 784 patches. |
| Seleção de amostras | Shuffle uniforme das imagens, seguido de permutação dos patches do lote. | WeightedRandomSampler com reposição e pesos inversos à frequência. | Dar massa de probabilidade igual aos grupos com/sem pluma, evitando domínio de patches vazios. |
| Distribuição esperada | Predominância natural de fundo. | Aproximadamente 50% de cada grupo em expectativa. | Mesma estratégia oficial de balanceamento. Não garante metade de cada classe em cada batch. |
| Critério has_plume | Não era usado para amostrar. | Fração de pixels positivos > 10/64²; em 128×128, mais de 40 pixels. | Copiar o critério oficial, em vez de contar qualquer pixel isolado como patch positivo. |
| Leitura prévia das máscaras | Sem estatística de todos os patches. | Calcula frac_positives e has_plume antes de treinar. | Obter os pesos necessários ao sampler. Essa etapa é CPU/I/O. |
| Validação dos arquivos | Conferia mag1c por padrão. | Confere todos os produtos de entrada, labelbinary e weight_mag1c. | Evitar aceitar amostras incompletas que falhariam durante o treino. |
| Workers | 4 fixos. | Padrão 4, configurável. | Manter a referência oficial e permitir ajuste ao host. |
| Pin memory | Sempre ativada. | Ativada quando o dispositivo é CUDA. | Preservar a transferência não bloqueante sem fixar memória para execução exclusivamente CPU. |
| Loss BCE | BCE por pixel × weight_mag1c, média. | Mesma fórmula, com pos_weight configurável, padrão 1. | Reproduzir a loss do HyperSTARCOP mag1c+RGB sem retirar a opção de ajuste. |
| Focal Dice | Alpha 0,25, gamma 2, pesos focal/dice iguais a 1, fixos. | Mesmos padrões, agora configuráveis. | Preservar a loss alternativa do projeto. Ela não é a loss original do artigo. |
| Ordem das augmentações | Flip horizontal, vertical, rotação. | Rotação, flip horizontal, vertical. | Seguir a sequência oficial. Probabilidades de 0,5 e rotação ±90° permanecem. |
| Transformação do mapa de pesos | Tratado como mask. | Tratado como input; gabarito continua mask. | Interpolar pesos contínuos como no oficial, mantendo a máscara binária com interpolação de máscara. |
| Otimizador | Adam, LR 1e-4 fixo no código. | Adam, LR inicial 1e-4 configurável. | Preservar o otimizador oficial e expor o parâmetro. |
| Scheduler | Ausente. | ReduceLROnPlateau, fator 0,5, paciência 4, monitorando val_loss; atualização ao fim da época. | Reduzir LR quando a loss de validação estaciona, como no oficial. |
| Limite de épocas | 200. | Padrão 15, configurável. | Reproduzir a duração configurada nos comandos do artigo. |
| Precisão | AMP ativado em CUDA. | FP32 por padrão; AMP opcional. | O Trainer oficial consultado não solicita precisão mista; manter AMP como opção local. |
| Escolha do dispositivo | CUDA se disponível, senão CPU. | auto, cuda ou cpu; CUDA solicitada e ausente gera erro. | Preservar a seleção automática e permitir escolha explícita no notebook. |
| Resolução de validação | Patches 128×128. | Imagens completas 512×512. | Reproduzir o datamodule oficial. |
| Batch de validação | Mesma quantidade de imagens do treino, depois achatadas em patches. | val_batch_size independente; padrão 32 imagens completas. | Corresponder ao oficial e permitir redução se faltar VRAM. |
| Frequência da validação | Uma vez no fim da época. | A cada fração configurável; padrão 0,5, com verificação ao final. | Seguir a configuração oficial e avaliar durante a época. Por arredondamento, um número ímpar de batches pode produzir uma verificação extra próxima ao final. |
| Loss de validação | Não calculada. | Mesma loss escolhida no treino, com mapa de pesos. | Produzir o critério de checkpoint/scheduler oficial. |
| F1 de validação | Média do F1 individual de cada patch. | F1 global por TP/FP/FN acumulados, informativo. | Evitar que muitos patches vazios gerem uma média alta sem detecção de pluma. |
| Pós-processamento na validação | Abertura morfológica em cruz 3×3. | Nenhuma abertura na validação de treinamento. | Corresponder ao model_module oficial. O pós-processamento do script de testes não foi alterado. |
| Threshold da validação | Logits > 0. | Logits >= 0. | Seguir a condição usada no treinamento oficial. |
| Checkpoint | Maior F1 médio por patch. | Menor val_loss. | Reproduzir o critério oficial e deixar de salvar pela média inflada por patches vazios. |
| Early stopping | Ativo, 10 épocas sem melhorar F1. | Desligado por padrão; opcional, paciência 8 verificações de validação sem melhorar loss. | O script oficial cria o callback, mas não o inclui na lista usada pelo Trainer. Não foi tratado como se estivesse ativo. |
| starting_point | F1 inicial a superar. | Argumento aceito por compatibilidade, sem controlar val_loss. | Um F1 não pode ser usado como limite inicial de uma loss. |
| Retomada de pesos | Carregava pesos e começava com starting_point. | Carrega pesos e mede a loss de validação inicial. | Só substituir os pesos retomados quando houver melhora nessa mesma execução. Não restaura otimizador/scheduler. |
| Média da loss de treino | Média simples dos batches. | Ponderada pelo número de amostras. | Dar o peso correto ao último batch, que pode ser menor. |
| Valores inválidos | Sem verificação explícita. | Interrompe ao encontrar loss não finita no treino ou validação. | Evitar continuar e salvar artefatos inválidos. É uma proteção local. |
| Histórico | Saídas do notebook. | Saídas e arquivo .training.json com configurações, loss, F1, LR e checkpoints. | Tornar as decisões de treinamento rastreáveis; registro local, sem exigir WandB. |
| Retorno da função | Nenhum. | Caminho do checkpoint, melhor val_loss e histórico. | Permitir consultar o resultado de maneira programática. |
| Configuração no notebook | Valores espalhados e alguns parâmetros embutidos nas chamadas. | LOSS_TREINO e CONFIG_TREINO centralizados. | Preservar as escolhas existentes e facilitar repetir experimentos. |
| Nomes dos novos pesos | Nomes anteriores. | Sufixo _hyperstarcop_bce ou _hyperstarcop_focal_dice. | Separar o novo protocolo dos checkpoints antigos. |
| Testes no notebook | Alguns nomes antigos ou fixos. | Nomes correspondentes ao modelo, protocolo e loss configurados. | Testar os pesos produzidos pelas células de treino atuais. |
| Saídas antigas das células alteradas | Resultados de chamadas anteriores. | Limpas na atualização do notebook. | Evitar apresentar um resultado antigo como se fosse do novo protocolo. |

## O que foi preservado e diferenças deliberadas

- O modelo continua sendo a arquitetura selecionada em MODELS. O HyperSTARCOP original usa U-Net de segmentation_models_pytorch com encoder MobileNetV2; este projeto continua permitindo MobileNetV3, attention gates e os demais modelos.
- As arquiteturas e seus valores de align_corners não foram alterados na adaptação do protocolo. A MobileNetV3 normal já tinha align_corners=False da alteração anterior.
- Normalização e ordem dos canais continuam consistentes entre treino e teste no projeto. O oficial mag1c+RGB usa mag1c, 640nm, 550nm, 460nm; o notebook usa mag1c, 460nm, 550nm, 640nm.
- A função antiga calcular_f1_score continua disponível por compatibilidade, mas não decide o checkpoint.
- Checkpoints continuam sendo state_dict em .pth, compatíveis com Teste_Unet.py; não são checkpoints completos do Lightning.
- O F1 acumulado local retorna 0 quando seu denominador é zero. As fórmulas oficiais podem retornar NaN em casos indefinidos. Nenhum desses F1 controla o checkpoint agora.
- A função mantém loss_name="focal_dice" por compatibilidade. O notebook configura explicitamente LOSS_TREINO="bce" para o protocolo do HyperSTARCOP.
- Os controles de CPU/CUDA, AMP, loss alternativa, split e proteção contra loss inválida são opções locais preservadas/adicionadas. Portanto, a configuração padrão do protocolo é alinhada ao oficial, mas usar essas alternativas muda o experimento.

## Como usar no main.ipynb

1. Reinicie o kernel após editar o módulo de treinamento.
2. Execute as importações.
3. Selecione o modelo em MODELS e ajuste LOSS_TREINO/CONFIG_TREINO.
4. Execute apenas a célula do conjunto de canais desejado.

```python
LOSS_TREINO = "bce"  # ou "focal_dice"
CONFIG_TREINO = dict(
    loss_name=LOSS_TREINO,
    resume=False,
    batch_size=32,        # patches 128×128
    num_workers=4,
    val_batch_size=32,    # imagens completas 512×512
    epocas=15,
    lr=1e-4,
    pos_weight=1.0,
    weight_sampling=True,
    validation_mode="official",  # "split" reserva 15%, seed 42
    val_check_interval=0.5,
    early_stopping=False,
    paciencia_maxima=8,
    lr_decay=0.5,
    lr_patience=4,
    device="cuda",
    amp=False,
    focal_alpha=0.25,
    focal_gamma=2.0,
    weight_focal=1.0,
    weight_dice=1.0,
)
SUFIXO_TREINO = "_hyperstarcop_" + LOSS_TREINO
```

O restante da célula de configuração do notebook já fornece esses controles. Ao alterar LOSS_TREINO, reexecute a célula inteira para atualizar o dicionário e o sufixo. Um batch de validação menor reduz a VRAM; com Focal Dice, seu tamanho também pode mudar a loss agregada, pois a Dice é calculada por lote.

No modo official, test.csv participa da seleção de pesos e do scheduler. Portanto, uma avaliação posterior nesse mesmo conjunto não é independente da seleção do modelo. Use validation_mode="split" se quiser reservar test.csv para a avaliação final.

## Diagnóstico de baixa utilização da RTX 4070 Super — 01/10/2026

O treinamento observado estava na primeira época, com BCE, batch de 32 patches, 4 workers e device CUDA. A GPU estava de fato em uso pelo kernel Jupyter. A baixa ocupação foi acompanhada de CPU ocupada pelos leitores:

- Em 15 amostras separadas por cerca de 1 segundo, a utilização total da GPU variou de 6% a 47%, com aproximadamente 2,2 GB de VRAM total ocupada, incluindo o desktop. O processo de treino aparecia com aproximadamente 780 MiB na consulta inicial.
- Os quatro workers ficaram próximos de 100% de um núcleo cada. Em cerca de 15 segundos, cada worker leu aproximadamente 2,5 GB pela interface de arquivos. O contador read_bytes não cresceu: esses dados já estavam no cache do sistema, não necessariamente sendo buscados do NVMe físico.
- O dataset está no NVMe /dev/nvme1n1p1, em ext4.
- Os arquivos inspecionados de mag1c, RGB e labelbinary usam LZW e blocos de 512×512; weight_mag1c usa blocos de 64×64. Ler uma janela 128×128 de um bloco comprimido 512×512 demanda descomprimir o bloco maior.
- O STARCOPDataset atual abre novamente os seis TIFFs de cada patch: quatro canais, máscara e mapa de pesos. Um batch de 32 provoca 192 aberturas. Antes, a leitura inteira de uma imagem era aproveitada para produzir seus 49 patches.
- Um ensaio exploratório, durante o treino, mediu 0,583 s para ler 32 patches (192 aberturas) e 0,187 s para ler 8 imagens inteiras (48 aberturas), usando os mesmos oito conjuntos de TIFFs. As quantidades de pixels retornados são diferentes; isso ilustra o custo das aberturas/descompressões e não é uma comparação isolada de throughput nem uma promessa de ganho.

**Interpretação:** a CPU fica ocupada abrindo e descomprimindo arquivos enquanto a GPU aguarda os próximos dados. O batch menor também fornece menos trabalho por atualização: 32 patches em vez de 196 na versão imediatamente anterior, ou 784 nos dois experimentos antigos. O código ainda converte a loss CUDA para números Python a cada passo, provocando sincronizações; isso é um fator adicional, não o principal gargalo medido.

### Ajustes possíveis sem mudar a amostragem e o batch oficial

- Experimentar num_workers=8, acompanhando tempo por batch e uso de CPU. Mais workers podem ajudar até saturar CPU, memória ou I/O; não há garantia de que mais sempre seja melhor.
- Implementar cache limitado de TIFFs/dados decodificados por worker, ou preparar os patches em armazenamento adequado. Isso pode manter os mesmos patches, pesos e probabilidades do sampler sem redescomprimir o mesmo bloco repetidamente. Ainda não foi implementado nesta análise.
- Considerar persistent_workers e prefetch_factor para manter leitores preparados entre épocas/validações. Essas opções foram implementadas na atualização descrita abaixo; a medição acima é anterior à sua ativação.
- Reduzir conversões GPU→Python e frequência de atualização do tqdm em uma futura otimização de execução, preservando a fórmula da loss.

Aumentar batch_size pode elevar a ocupação da GPU, mas altera o tamanho efetivo de batch e deixa de reproduzir o padrão de 32 patches do oficial. amp=True é outra opção já disponível para throughput/VRAM, mas modifica a precisão e não resolve sozinho a leitura lenta de TIFFs. Não foram alterados os parâmetros nem interrompido o treino durante esse diagnóstico.

O uso de VRAM não precisa chegar ao limite para que o treinamento esteja correto. A medida útil é patches processados por segundo e tempo por época; porcentagem de GPU isoladamente não mede qualidade do modelo.

## Artefatos e verificação

Os pesos são gravados em Modelos_treinados/<nome>.pth e o histórico em <nome>.training.json. O histórico registra qual verificação salvou o checkpoint. resume=True carrega só pesos; não é retomada exata de uma sessão anterior.

A implementação foi validada com dados sintéticos nas duas losses: batch independente, massa de amostragem balanceada, gradientes, validação em 512×512, checkpoint pela menor loss e registro de histórico. Também foram conferidas as formas de saída da MobileNetV3 em 128×128 e 512×512. Esses testes verificam o fluxo de execução, não garantem métricas de um treino completo.

Veja também [PROTOCOLO_HYPERSTARCOP.md](PROTOCOLO_HYPERSTARCOP.md). Depois do treinamento, siga a [documentação Vitis AI](../VitisAI/README.md) para quantizar/exportar/compilar; esse fluxo não substitui o treinamento.


## Atualização de carregamento: cache, prefetch e workers persistentes

Implementada após o diagnóstico, sem mudar AMP, tqdm, batch, loss, amostragem ou arquitetura:

| Controle no notebook | Configuração | Funcionamento |
|---|---|---|
| num_workers | 8, escolha atual do usuário | Oito leitores para treino e oito para validação. |
| prefetch_factor | 4 | Até quatro batches antecipados por worker. Só é passado ao DataLoader quando num_workers > 0. |
| persistent_workers | True | Mantém os processos vivos e seus caches entre épocas e verificações de validação. Com num_workers=0, é omitido automaticamente. |
| cache_max_gb | 2,0 | Limite de 2 GiB de arrays TIFF descomprimidos por worker de treino. |
| val_cache_max_gb | 0,25 | Limite separado de 256 MiB por worker de validação. |

O cache é LRU: remove os arquivos usados há mais tempo para respeitar o teto em bytes. Guarda arrays descomprimidos dos TIFFs e copia somente o recorte solicitado para cada amostra. Não mantém arquivos abertos. Cada worker possui seu próprio cache; o limite é aplicado antes de carregar outro arquivo completo. Arquivos maiores que o teto, janelas fracionárias ou fora dos limites seguem a leitura Rasterio original. O cache é desativável com limite 0 e fica desativado por padrão no STARCOPDataset usado fora do treinamento.

Com oito workers por loader, o teto dos arrays em cache é 8×2 GiB + 8×0,25 GiB = **18 GiB de RAM**, atingido apenas se os caches encherem. Esse teto não inclui batches em filas, arrays temporários, metadados, memória do Python, memória fixada ou cache do sistema operacional. Com batch de treino 32 e seis bandas float32 em 128×128, as filas de quatro batches por worker podem conter cerca de 384 MiB de tensores; com validação de 32 imagens 512×512, cerca de 6 GiB. São estimativas, não tetos da RAM total do processo. A configuração foi dimensionada considerando 64 GB de RAM e a GPU de 12 GB de VRAM observada no host; esses caches usam RAM, não VRAM.

O cache por worker pode duplicar uma imagem entre leitores. A amostragem aleatória em um dataset grande também pode limitar a taxa de acertos: não há promessa de ganho fixo. Workers persistentes permitem aquecer os caches ao longo do treino. O histórico .training.json passa a registrar os quatro novos controles.

As mudanças só entram em uma nova chamada de treinamento após recarregar os módulos; a execução que já está rodando continua com os DataLoaders antigos. Reinicie o kernel antes de iniciar a próxima execução. Não reinicie um treino em andamento sem encerrar conscientemente sua execução. O cache considera os arquivos imutáveis durante a sessão; após substituir TIFFs, crie novos datasets/workers.

Verificação realizada: igualdade exata entre leituras com/sem cache nos quatro canais, máscara e pesos; limite de bytes e expulsão LRU; isolamento contra alterações na amostra; janelas fracionárias e externas; serialização dos workers; persistência dos mesmos processos em múltiplas épocas e prefetch=4. Nenhum checkpoint real foi alterado pelos testes.

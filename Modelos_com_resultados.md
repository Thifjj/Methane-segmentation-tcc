## Modelos já com resultados (CPU/GPU, ARM e ZCU104)

- HYPERSTARCOP_OFICIAL
- UNET_MOBILENET_V3_ATTENTIONGATES_DPU_BCE_REMAINING_ALL
- UNET_MOBILENET_V3_ATTENTIONGATES_DPU_BCE_RETRAINED -- pode ser excluida
- UNET_MOBILENET_V3_ATTENTIONGATES_DPU_FOCALDICE_REMAINING_ALL
- UNET_MOBILENET_V3_BCE_DPU
- UNET_MOBILENET_V3_FOCALDICE_DPU

Os checkpoints BCE `remaining_all` e `retrained` são diferentes. Os resultados
históricos devem ser associados ao checkpoint usado em cada execução.

## Modelos SegFormer pendentes de quantização e benchmarks

- SEGFORMER_B0_BCE
- SEGFORMER_B0_FOCALDICE
- SEGFORMER_B0_ATTENTIONGATES_BCE
- SEGFORMER_B0_ATTENTIONGATES_FOCALDICE

Os artefatos e as avaliações de quantização anteriores foram removidos porque
eram testes antigos. Esses resultados não contam como etapas concluídas.
Os checkpoints FP32 permanecem disponíveis; faltam quantização, avaliação INT8,
compilação para DPU e benchmarks ARM/ZCU104.

A variante `SEGFORMER_B0_ATTENTIONGATES` sem identificação de loss foi retirada
desta lista conforme a seleção atual; seu checkpoint não foi alterado.

## Modelos com resultados parciais

- UNET_MOBILENET_V3_BCE — resultados CPU; a variante DPU está na primeira lista.
- UNET_MOBILENET_V3_FOCALDICE — resultados com a classe DPU; não confirmar a classe padrão apenas pelos pesos.
- UNET_MOBILENET_V3_ATTENTIONGATES_BCE — resultados CPU.
- UNET_MOBILENET_V3_ATTENTIONGATES_FOCALDICE — resultados CPU e ARM.
- UNET_MOBILENET_V3_ATTENTIONGATES_FOCALDICE_REMAINING_ALL — resultados CPU.
- UNET_MOBILENET_V3_ATTENTIONGATES_DPU_ONLY_REMAINING_VAL_TEST — resultados CPU e avaliações FP32/INT8.
- UNET_MOBILENET_V3_ATTENTIONGATES_DPU_EASY_REMAINING — resultados CPU e avaliações FP32/INT8; checkpoint ausente neste checkout.

## Modelos treinados sem resultados de benchmark confirmados

- UNET_BASELINE
- UNET_DEPTH_REDUCED
- UNET_SKIP_CONNECTIONS
- UNET_MOBILENET_V2
- UNET_RESNET34
- UNET_MOBILENET_V3_ATTENTIONGATES_BCE_OLD — checkpoint distinto, sem resultados associados confirmados. -- pode remover

Há checkpoints disponíveis para esses modelos. Ainda é necessário confirmar ou
executar os benchmarks e, para o fluxo DPU, verificar quantização e compilação.
`UNET_SKIP_CONNECTIONS` foi retirado da lista de concluídos: não foram encontrados
arquivos de benchmark que confirmem essa conclusão.

## Modelos sem checkpoint treinado disponível

- UNET_ATTENTIONGATES
- UNET_PSA

A arquitetura existe em `Modelos/`, mas não há checkpoint correspondente no
registro atual. Isso não comprova que nunca tenham sido treinados fora deste checkout.

## Referências

- [Registro dos checkpoints e nomes padronizados](Modelos_treinados/MODEL_REGISTRY.md).
- [Resultados CPU/GPU do artigo](benchmark_nao_embarcado/resultados_artigo/) e [resultados CPU](benchmark_nao_embarcado/resultados_cpu/).
- [Resultados ARM](benchmark_arm/resultados_arm/).
- [Resultados ZCU104](benchmark_zcu104/resultados_zcu104/).

FD significa Focal Dice. Os nomes acima identificam modelos e variantes;
os nomes exatos dos arquivos e seus hashes estão no registro dos checkpoints.

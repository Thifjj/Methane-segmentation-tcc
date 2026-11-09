# Acessar o Vitis AI 3.5 — Projeto de segmentação de metano

Este guia reutiliza o contêiner Docker já configurado (`methane-vitis-ai-cpu`). **Não cria nem reinstala o ambiente.**

## 1. Iniciar o contêiner

No terminal do **Kubuntu (fora do Docker)**:

```bash
docker start methane-vitis-ai-cpu
```

> Se o contêiner já estiver em execução, prossiga para a próxima etapa.

## 2. Entrar no contêiner

```bash
docker exec -it methane-vitis-ai-cpu bash
```

## 3. Ativar o ambiente PyTorch do Vitis AI

Dentro do contêiner:

```bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
```

O prompt deverá começar com `(vitis-ai-pytorch)`.

## 4. Abrir o projeto

```bash
cd /workspace/VitisAI
export PYTHONPATH=/workspace:$PYTHONPATH
```

## 5. Verificar a instalação

```bash
python -c "import torch, pytorch_nndct; print('PyTorch:', torch.__version__); print('Vitis AI pronto')"
```

Resultado esperado: `PyTorch: 1.13.1` e `Vitis AI pronto`.

## 6. Conferir arquivos do projeto (opcional)

```bash
ls /workspace/Modelos_treinados/
ls /workspace/VitisAI/
ls /workspace/VitisAI/build/vitis_ai/calibration_sets/
```

## 7. Sair do ambiente

Para sair do contêiner sem apagá-lo:

```bash
exit
```

Para desligá-lo **depois de sair**, no terminal do Kubuntu:

```bash
docker stop methane-vitis-ai-cpu
```

**Atenção:** não execute `docker rm`, `docker system prune` ou comandos que removam volumes. O contêiner e os dados do projeto devem ser preservados.

## Acesso rápido (próximas sessões)

No terminal do Kubuntu:

```bash
docker start methane-vitis-ai-cpu
docker exec -it methane-vitis-ai-cpu bash
```

Dentro do contêiner:

```bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
cd /workspace/VitisAI
export PYTHONPATH=/workspace:$PYTHONPATH
```

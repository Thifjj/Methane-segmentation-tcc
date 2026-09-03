import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from kornia.morphology import erosion, dilation
import os

from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer

def binary_opening(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    eroded = torch.clamp(erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(dilation(eroded.float(), kernel), 0, 1) > 0

def calcular_f1_score(previsao_logits, gabarito, threshold=0.0):
    device = previsao_logits.device
    previsao_binaria = (previsao_logits > threshold).float()
    
    kernel_cruz = torch.tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]).float().to(device)
    
    previsao_limpa = binary_opening(previsao_binaria, kernel_cruz).float()
    
    intersecao = (previsao_limpa * gabarito).sum(dim=(2, 3))
    soma_areas = previsao_limpa.sum(dim=(2, 3)) + gabarito.sum(dim=(2, 3))
    f1 = (2 * intersecao + 1e-6) / (soma_areas + 1e-6)
    return f1.mean().item()

def treinar_modelo(modelo_escolhido,nome_modelo_salvar, produtos_entrada):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Iniciando Treinamento: {nome_modelo_salvar} ---")
    print(f"Bandas utilizadas: {produtos_entrada}\n")


    CAMINHO_CSV = "/media/jacques/hdd/Laboratorio/2_2026_lab/Projeto_pesquisaMetano/dataset_STARCOP/train.csv"
    DIRETORIO_DADOS = "/media/jacques/hdd/Laboratorio/2_2026_lab/Projeto_pesquisaMetano/dataset_STARCOP"
    PRODUTO_SAIDA = ["labelbinary"]

    print(f"Dataset: {DIRETORIO_DADOS}\n")
    
    df_train = carregar_dataframe_starcop(CAMINHO_CSV, DIRETORIO_DADOS)
    dataset_treino = STARCOPDataset(df_train, produtos_entrada, PRODUTO_SAIDA, weight_loss="weight_mag1c")
    dataloader = DataLoader(
        dataset_treino,
        batch_size=6,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )
    normalizador = DataNormalizer(produtos_entrada).to(device)

    modelo = modelo_escolhido(in_channels=len(produtos_entrada), out_channels=1).to(device)

    caminho_salvamento = f"Modelos_treinados/{nome_modelo_salvar}.pth"
    
    # Garante que a pasta 'Modelos_treinados' exista, se não, ele a cria para evitar erro no salvamento
    os.makedirs("Modelos_treinados", exist_ok=True)
    
    if os.path.exists(caminho_salvamento):
        print(f"Encontrado modelo treinado com o mesmo nome: '{caminho_salvamento}'")
        print("Carregando pesos para retomar o treinamento de onde paro\n")
        print("Caso queira iniciar um treinamento do zero, exclua o modelo em 'Modelos_treinados' ou passe outro nome como parametro\n")
        modelo.load_state_dict(torch.load(caminho_salvamento, map_location=device, weights_only=True))
    
    optimizer = optim.Adam(modelo.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    epocas = 200
    paciencia_maxima = 10 
    melhor_f1 = 0.0
    paciencia_atual = 0

    for epoca in range(epocas):
        modelo.train()
        f1_acumulado = 0.0
        
        loop = tqdm(dataloader, desc=f"Época {epoca+1}/{epocas}")
        for batch in loop:
            inputs = normalizador(batch["input"].to(device, non_blocking=True))
            targets = batch["output"].to(device, non_blocking=True)
            pesos_loss = batch["weight_loss"].to(device, non_blocking=True)

            optimizer.zero_grad()
            previsoes = modelo(inputs)

            loss = (criterion(previsoes, targets) * pesos_loss).mean()
            loss.backward()
            optimizer.step()

            f1_atual = calcular_f1_score(previsoes, targets)
            f1_acumulado += f1_atual
            loop.set_postfix(Loss=f"{loss.item():.4f}", F1=f"{f1_atual:.3f}")

        media_f1 = f1_acumulado / len(dataloader)
        
        if media_f1 > melhor_f1:
            melhor_f1 = media_f1
            paciencia_atual = 0
            torch.save(modelo.state_dict(), f"Modelos_treinados/{nome_modelo_salvar}.pth")
            print(f" -> Novo recorde F1: {melhor_f1:.4f}. Modelo salvo!")
        else:
            paciencia_atual += 1
            if paciencia_atual >= paciencia_maxima:
                print(f"=== Early Stopping! Melhor F1-Score: {melhor_f1:.4f} ===")
                break

if __name__ == "__main__":

    print("Use as funções através do documento main.ipynb")

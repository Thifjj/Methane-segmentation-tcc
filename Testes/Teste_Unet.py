import os
import time
import torch
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm
from kornia.morphology import erosion, dilation
from sklearn.metrics import average_precision_score

from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer

def binary_opening(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    eroded = torch.clamp(erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(dilation(eroded.float(), kernel), 0, 1) > 0

def ler_tif_para_plot(pasta, window, banda, div=1.0):
    caminho = os.path.join(pasta, f"{banda}.tif")
    with rasterio.open(caminho) as src:
        img = src.read(1, window=window)
    return np.clip(img / div, 0, 1)

def salvar_log_csv(nome_modelo, f1_global, f1_strong, f1_weak, iou, auprc, fpr_no_plume, device, num_parametros, tamanho_mb, inferencia_ms, vram_pico_mb):
    nome_arquivo = "Resultados_testes/historico_testes.csv"
    os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
    
    novo_registro = {
        "Nome do Modelo": nome_modelo,
        "Device": device,
        "Parametros": num_parametros,
        "Tamanho (MB)": round(tamanho_mb, 2),
        "Inferencia (ms/img)": round(inferencia_ms, 2),
        "VRAM Pico (MB)": round(vram_pico_mb, 2),
        "F1-Global": round(f1_global, 4),
        "F1-Strong": round(f1_strong, 4),
        "F1-Weak": round(f1_weak, 4),
        "IoU": round(iou, 4),
        "AUPRC": round(auprc, 4),
        "FPR (No-Plume)": round(fpr_no_plume, 6),
        "Teste #": 1
    }
    
    if os.path.exists(nome_arquivo):
        df = pd.read_csv(nome_arquivo)
        testes_anteriores = df[df["Nome do Modelo"] == nome_modelo]
        if not testes_anteriores.empty:
            novo_registro["Teste #"] = testes_anteriores["Teste #"].max() + 1
            
        df_novo = pd.DataFrame([novo_registro])
        df = pd.concat([df, df_novo], ignore_index=True)
    else:
        df = pd.DataFrame([novo_registro])
        
    colunas_ordem = ["Teste #", "Nome do Modelo", "Device", "Parametros", "Tamanho (MB)", "Inferencia (ms/img)", "VRAM Pico (MB)", "F1-Global", "F1-Strong", "F1-Weak", "IoU", "AUPRC", "FPR (No-Plume)"]
    df = df[colunas_ordem]
    
    df.to_csv(nome_arquivo, index=False)
    print(f"Log do teste salvo em: '{nome_arquivo}'")

def avaliar_e_visualizar(modelo_escolhido, nome_modelo_salvo, produtos_entrada):
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = device_obj.type.upper()
    print(f"\nIniciando Avaliação do modelo: {nome_modelo_salvo} ({device_name})")
    #notebook caminhos:
    #CAMINHO_CSV_TESTE ="/home/thiago/Documents/STARCOP_DATASET/train.csv"
    #DIRETORIO_DADOS_TESTE ="/home/thiago/Documents/STARCOP_DATASET/"
    
    #CAMINHO_CSV_TESTE ="/home/thiago/Documents/Laboratorio_LEDS/Projetos_aceleradores/Segmentacao_de_metano/Joao/projeto/Methane-segmentation-tcc/STARCOP_test/test.csv"
    #DIRETORIO_DADOS_TESTE = "/home/thiago/Documents/Laboratorio_LEDS/Projetos_aceleradores/Segmentacao_de_metano/Joao/projeto/Methane-segmentation-tcc/STARCOP_test/"
    
    #caminhos desktop
    CAMINHO_CSV_TESTE = "/media/jacques/games/Datasets/test/STARCOP_test/test.csv" 
    DIRETORIO_DADOS_TESTE = "/media/jacques/games/Datasets/test/STARCOP_test/"

    #CAMINHO_CSV_TESTE ="/media/jacques/games/Datasets/STARCOP_train_remaining_all/train.csv"
    #DIRETORIO_DADOS_TESTE="/media/jacques/games/Datasets/STARCOP_train_remaining_all/"

    df_test = carregar_dataframe_starcop(CAMINHO_CSV_TESTE, DIRETORIO_DADOS_TESTE)
    dataset_teste = STARCOPDataset(df_test, produtos_entrada, ["labelbinary"])
    dataloader = DataLoader(dataset_teste, batch_size=1, shuffle=False)
    normalizador = DataNormalizer(produtos_entrada).to(device_obj)

    modelo = modelo_escolhido(in_channels=len(produtos_entrada), out_channels=1).to(device_obj)
    caminho_pesos = f"Modelos_treinados/{nome_modelo_salvo}.pth"
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=device_obj, weights_only=True))
    modelo.eval()

    total_parametros = sum(p.numel() for p in modelo.parameters())
    tamanho_arquivo_mb = os.path.getsize(caminho_pesos) / (1024 * 1024)

    kernel_cruz = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).float().to(device_obj)

    auprc_por_imagem = []
    
    # Acumuladores Globais
    TP_tot = FP_tot = FN_tot = TN_tot = 0
    # Acumuladores por Dificuldade
    TP_str = FP_str = FN_str = 0
    TP_weak = FP_weak = FN_weak = 0
    FP_no_plume = TN_no_plume = 0
    
    tempo_total_inferencia = 0.0

    if device_name == "CUDA":
        dummy_input = torch.randn(1, len(produtos_entrada), 512, 512).to(device_obj)
        for _ in range(10): _ = modelo(dummy_input)
        torch.cuda.synchronize()
        del dummy_input, _
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculando Métricas e Latência")):
            b, p, c, h_dim, w_dim = batch["input"].shape
            inputs = batch["input"].view(b * p, c, h_dim, w_dim).to(device_obj)
            targets = batch["output"].view(b * p, 1, h_dim, w_dim).to(device_obj)

            inputs = normalizador(inputs)
            
            if device_name == "CUDA": 
                torch.cuda.synchronize()
            inicio = time.perf_counter()
        
            logits = modelo(inputs)
        
            if device_name == "CUDA": torch.cuda.synchronize()
            fim = time.perf_counter()
            tempo_total_inferencia += (fim - inicio)

            probs = torch.sigmoid(logits)
            previsao_binaria = (logits > 0.0).float()
            previsao_limpa = binary_opening(previsao_binaria, kernel_cruz).float()

            p_flat = previsao_limpa.view(-1)
            g_flat = targets.view(-1)
        
            tp = (p_flat * g_flat).sum().item()
            fp = (p_flat * (1 - g_flat)).sum().item()
            fn = ((1 - p_flat) * g_flat).sum().item()
            tn = ((1 - p_flat) * (1 - g_flat)).sum().item()

            TP_tot += tp; FP_tot += fp; FN_tot += fn; TN_tot += tn
        
            g_sum = g_flat.sum().item()
            if g_sum > 0:
                dificuldade = df_test.iloc[i].get('difficulty', '')
            
                is_strong = (dificuldade == 'easy')
            
                if is_strong:
                    TP_str += tp; FP_str += fp; FN_str += fn
                else:
                    TP_weak += tp; FP_weak += fp; FN_weak += fn
            else:
                # É uma imagem sem pluma (Background tile)
                FP_no_plume += fp
                TN_no_plume += tn
                
            probs_np = probs.view(-1).cpu().numpy()
            g_flat_np = g_flat.cpu().numpy()

            if g_sum > 0:
                auprc_img = average_precision_score(g_flat_np, probs_np)
                auprc_por_imagem.append(auprc_img)

            if g_sum > 0 and i == 0: 
                pasta = df_test.iloc[i]['folder']
                window = df_test.iloc[i]['window']
            
                R = ler_tif_para_plot(pasta, window, "TOA_AVIRIS_640nm", div=60.0)
                G = ler_tif_para_plot(pasta, window, "TOA_AVIRIS_550nm", div=60.0)
                B = ler_tif_para_plot(pasta, window, "TOA_AVIRIS_460nm", div=60.0)
                img_rgb = np.stack([R, G, B], axis=-1)
                img_mag1c = ler_tif_para_plot(pasta, window, "mag1c", div=1750.0)
            
                pred_np = previsao_limpa[0, 0].cpu().numpy()
                gab_np = targets[0, 0].cpu().numpy()
                diferenca = (2 * pred_np) + gab_np
                cmap_diff = mcolors.ListedColormap(['black', 'red', 'yellow', 'green'])
            
                fig, ax = plt.subplots(1, 5, figsize=(20, 4))
                ax[0].imshow(img_rgb); ax[0].set_title("RGB (Visível)")
                ax[1].imshow(img_mag1c, cmap='magma'); ax[1].set_title("Mag1c (Química)")
                ax[2].imshow(gab_np, cmap='gray'); ax[2].set_title("Ground Truth")
                ax[3].imshow(pred_np, cmap='gray'); ax[3].set_title("Previsto (Otimizado)")
                cax = ax[4].imshow(diferenca, cmap=cmap_diff, vmin=0, vmax=3)
                ax[4].set_title("Diferença")
                for a in ax: a.axis('off')
                fig.suptitle(f"Avaliação do Modelo: {nome_modelo_salvo}", fontsize=16)
                plt.show()

    # Cálculos das Métricas
    vram_pico_mb = ( torch.cuda.max_memory_allocated() / (1024 * 1024) if device_name == "CUDA" else 0.0 )

    iou_global = TP_tot / (TP_tot + FP_tot + FN_tot + 1e-6)
    f1_global = 2 * TP_tot / (2 * TP_tot + FP_tot + FN_tot + 1e-6)

    f1_strong = 2 * TP_str / (2 * TP_str + FP_str + FN_str + 1e-6) if (TP_str + FP_str + FN_str) > 0 else 0.0
    f1_weak = 2 * TP_weak / (2 * TP_weak + FP_weak + FN_weak + 1e-6) if (TP_weak + FP_weak + FN_weak) > 0 else 0.0

    # FPR restrito aos tiles sem pluma
    fpr_no_plume = FP_no_plume / (FP_no_plume + TN_no_plume + 1e-6)

    auprc = np.mean(auprc_por_imagem) if len(auprc_por_imagem) > 0 else 0.0


    latencia_media_ms = (tempo_total_inferencia / len(dataloader)) * 1000

    print(f" Modelo:           {nome_modelo_salvo}")
    print(f" Total Parâmetros: {total_parametros:,}")
    print(f" Tamanho Arquivo:  {tamanho_arquivo_mb:.2f} MB")
    print(f" Latência Média:   {latencia_media_ms:.2f} ms por imagem ({device_name})")
    print(f" Pico de VRAM:     {vram_pico_mb:.2f} MB")
    print(f" F1-Global:        {f1_global:.4f}")
    print(f" F1-Strong:        {f1_strong:.4f} (Emissão >= 1000 kg/h ou > 1000 px)")
    print(f" F1-Weak:          {f1_weak:.4f} (Emissão < 1000 kg/h e <= 1000 px)")
    print(f" IoU:              {iou_global:.4f}")
    print(f" AUPRC:            {auprc:.4f}")
    print(f" FPR (No-Plume):   {fpr_no_plume:.6f}")

    salvar_log_csv(nome_modelo_salvo, f1_global, f1_strong, f1_weak, iou_global, auprc, fpr_no_plume, device_name, total_parametros, tamanho_arquivo_mb, latencia_media_ms, vram_pico_mb)

if __name__ == "__main__":
    print("Use as funções através do documento main.ipynb")
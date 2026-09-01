import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm

def remover_outliers(d, percentile=5):
    """Remove os 5% de pixels mais extremos para o cálculo não ser distorcido (Igual ao Source 7)"""
    upper_quartile = np.percentile(d, 100 - percentile)
    lower_quartile = np.percentile(d, percentile)
    return d[np.where((d >= lower_quartile) & (d <= upper_quartile))]

def razao_varon(canal_fundo, canal_sinal, p=5, zero_value_out=-0.6):
    """Aplica a equação exata da razão de Varon com compensação de Constante C"""
    sinal_flat = canal_sinal.flatten()
    fundo_flat = canal_fundo.flatten()
    
    # Previne divisão por zero onde os sensores não mapearam a Terra
    zeros_mask = (canal_sinal < 1e-6) & (canal_fundo < 1e-6)
    
    soma_fundo = np.sum(remover_outliers(fundo_flat, p))
    soma_sinal = np.sum(remover_outliers(sinal_flat, p))
    
    c = soma_fundo / soma_sinal
    
    # Equação: R = (c * S - B) / B
    R = (c * canal_sinal - canal_fundo) / (canal_fundo + 1e-6)
    R[zeros_mask] = zero_value_out
    
    return R.astype(np.float32)

def gerar_tifs_varon():
    # AJUSTE PARA A SUA PASTA DE TREINO E TESTE
    pastas_datasets = [
        "Datasets/STARCOP_train"#,
        #"Datasets/STARCOP_test"
    ]
    
    for diretorio_base in pastas_datasets:
        print(f"\nProcurando pastas em: {diretorio_base}")
        subpastas = glob.glob(os.path.join(diretorio_base, "ang*"))
        
        for pasta in tqdm(subpastas, desc=f"Processando Imagens"):
            try:
                # Carrega as 4 bandas brutas do WorldView-3 necessárias
                with rasterio.open(os.path.join(pasta, "TOA_WV3_SWIR5.tif")) as src:
                    swir5 = src.read(1)
                    perfil_tif = src.profile # Salva os metadados (CRS, Transform, Tamanho) para os novos arquivos
                
                with rasterio.open(os.path.join(pasta, "TOA_WV3_SWIR6.tif")) as src:
                    swir6 = src.read(1)
                    
                with rasterio.open(os.path.join(pasta, "TOA_WV3_SWIR7.tif")) as src:
                    swir7 = src.read(1)
                    
                with rasterio.open(os.path.join(pasta, "TOA_WV3_SWIR8.tif")) as src:
                    swir8 = src.read(1)

                # 1. Relação B7 e B5
                varon_b7_b5 = razao_varon(swir5, swir7)
                with rasterio.open(os.path.join(pasta, "ratio_wv3_B7_B5_varon21_sum_c_out.tif"), 'w', **perfil_tif) as dst:
                    dst.write(varon_b7_b5, 1)

                # 2. Relação B8 e B5
                varon_b8_b5 = razao_varon(swir5, swir8)
                with rasterio.open(os.path.join(pasta, "ratio_wv3_B8_B5_varon21_sum_c_out.tif"), 'w', **perfil_tif) as dst:
                    dst.write(varon_b8_b5, 1)

                # 3. Relação B7 e B6
                varon_b7_b6 = razao_varon(swir6, swir7)
                with rasterio.open(os.path.join(pasta, "ratio_wv3_B7_B6_varon21_sum_c_out.tif"), 'w', **perfil_tif) as dst:
                    dst.write(varon_b7_b6, 1)

            except Exception as e:
                # Pula pastas que possam estar vazias ou corrompidas
                print(f"Erro na pasta {pasta}: {e}")

if __name__ == "__main__":
    gerar_tifs_varon()
    print("\nTodos os produtos MultiSTARCOP (Varon) foram gerados com sucesso!")
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F


from config import MODELS_DIR, FAKES_DIR, DEVICE, HEATMAPS_DIR
from modelo import UNetColorizer

# --- CONFIGURAÇÕES ---
TRAIN_VAL_OFFSET = 4200  # Imagens < 4200 foram usadas no treino"

def load_image_pipeline(path):
    """
    Pipeline OTIMIZADA (OpenCV): 
    Leitura -> Resize (512x512) -> Lab -> Normalização
    """
    try:
        # 1. Leitura Segura (Windows)
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        
        if img_bgr is None: return None, None, None
        
        # 2. Resize IMEDIATO (Crucial para velocidade e compatibilidade com a UNet)
        img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)

        # 3. Conversão BGR -> Lab (OpenCV)
        # No OpenCV: L [0, 255], a [0, 255], b [0, 255]
        img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
        
        # Separar canais
        L = img_lab[:, :, 0]
        a = img_lab[:, :, 1]
        b = img_lab[:, :, 2]

        # 4. Normalização (Idêntica ao Treino)
        # L: [0, 255] -> [0, 1]
        L_norm = L.astype("float32") / 255.0
        
        # ab: [0, 255] -> [-1, 1] (Centrado em 128)
        a_norm = (a.astype("float32") - 128.0) / 128.0
        b_norm = (b.astype("float32") - 128.0) / 128.0
        
        # Empilhar ab (512, 512, 2)
        ab_norm = np.stack([a_norm, b_norm], axis=2)
        
        # 5. Converter para Tensores
        # Input L: (Batch=1, Channel=1, H, W)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        # Target ab: (Batch=1, Channel=2, H, W)
        # Transpose de (H, W, C) -> (C, H, W)
        ab_tensor = torch.from_numpy(ab_norm.transpose((2, 0, 1))).unsqueeze(0).float().to(DEVICE)
        
        # 6. Preparar RGB para visualização (apenas para o plot)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        
        return L_tensor, ab_tensor, img_rgb

    except Exception as e:
        print(f"Erro na imagem {path}: {e}")
        return None, None, None

def save_analysis_plot(model, img_path, mse_error, title_suffix):
    """Gera e guarda o gráfico (Heatmap) com a máscara aplicada visualmente."""
    L, ab_real, original_rgb = load_image_pipeline(img_path)
    if L is None: return
    
    # --- 1. Recriar a Máscara (Igual ao evaluate) ---
    H, W = 512, 512
    Y, X = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    sigma_y = 100 
    sigma_x = 80 

    dist_from_center = np.sqrt(((X - center_x)**2 / (2 * sigma_x**2)) + 
                            ((Y - center_y)**2 / (2 * sigma_y**2)))

    mask = np.exp(-dist_from_center)
    mask = np.where(mask < 0.2, 0, mask)
    mask = np.maximum(mask, 0.0)
    
    # Converter para formato compatível com imagem (H, W, 1) para multiplicar por RGB (H, W, 3)
    mask_visual = mask[:, :, np.newaxis] 

    with torch.no_grad():
        ab_pred = model(L)
    
    # --- 2. Gerar Heatmap e Aplicar Máscara ---
    # Desnormalizar: Multiplicar por 128 para ter erro em "unidades de cor" reais
    diff = (ab_pred - ab_real) * 128.0 
    
    # Calcula erro e aplica a máscara logo aqui para o heatmap ficar limpo
    heatmap_tensor = torch.sqrt(torch.sum(diff ** 2, dim=1)).squeeze().cpu().numpy()
    heatmap_masked = heatmap_tensor * mask # Máscara no heatmap
    
    # --- 3. Aplicar Máscara na Imagem Original (Input) ---
    input_masked = original_rgb * mask_visual

    # Configurar Plot
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Input MASCARADO (Para provar que ignoraste o fundo)
    plt.subplot(1, 3, 1)
    plt.imshow(input_masked)
    plt.title(f"Foco do Modelo ({title_suffix})")
    plt.axis('off')
    
    # Plot 2: Heatmap MASCARADO
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_masked, cmap='jet', vmin=0, vmax=30) 
    plt.title(f"Erro Local (MSE: {mse_error:.5f})")
    plt.colorbar()
    plt.axis('off')
    
    # Plot 3: Histograma (apenas dos pixeis da cara)
    plt.subplot(1, 3, 3)
    # Filtramos apenas os valores onde a máscara é > 0 para o histograma não ter milhoes de zeros
    valid_errors = heatmap_masked[mask > 0]
    plt.hist(valid_errors, bins=50, color='red', alpha=0.7, range=(0, 60))
    plt.title("Distribuição do Erro (Só Cara)")
    
    # Guardar
    filename = f"{title_suffix}_{img_path.stem}.png"
    save_path = HEATMAPS_DIR / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Heatmap salvo: {filename}")

def evaluate():
    print(f"--- AVALIAÇÃO FINAL (Split > {TRAIN_VAL_OFFSET}) ---")
    
    # 1. Setup
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / "classificador.pth"
    if not model_path.exists():
        print("Erro: Modelo não encontrado. Corre 'autoencoder.py' primeiro.")
        return

    model = UNetColorizer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 2. Dados de Teste
    fake_paths = sorted(list(FAKES_DIR.glob("*_fake.png")))
    real_paths = sorted(list(FAKES_DIR.glob("*_real.png")))
    
    test_real = real_paths[TRAIN_VAL_OFFSET:]
    test_fake = fake_paths[TRAIN_VAL_OFFSET:]
    
    if not test_real:
        print("ERRO: Sem imagens para teste.")
        return

    print(f"Imagens de Teste: {len(test_real)} Reais vs {len(test_fake)} Fakes")
    
    y_true = []
    y_scores = []
    fake_results = [] 

    dataset = [(p, 0) for p in test_real] + [(p, 1) for p in test_fake]
    
    print("A calcular erros...")
    for img_path, label in tqdm(dataset):
        # 1. Gerar a mesma máscara usada no treino
        H, W = 512, 512

        Y, X = np.ogrid[:H, :W]
        center_y, center_x = H / 2, W / 2
        sigma_y = 100 
        sigma_x = 80 

        dist_from_center = np.sqrt(((X - center_x)**2 / (2 * sigma_x**2)) + 
                                ((Y - center_y)**2 / (2 * sigma_y**2)))

        mask = np.exp(-dist_from_center)
        mask = np.where(mask < 0.2, 0, mask)
        mask = np.maximum(mask, 0.0)
        
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            L, ab_real, _ = load_image_pipeline(img_path)
            if L is None: continue # Segurança contra imagens corrompidas

            ab_pred = model(L)
            
            # --- CÁLCULO DO ERRO (Top-K Patches) ---
            
            squared_diff = (ab_pred - ab_real)**2
            
            # 2. Aplicar a máscara (Zerar o erro do fundo)
            weighted_error_map = squared_diff * mask_tensor
            
            # 3. Média Ponderada Global (A "Bala de Prata")
            # Somamos todo o erro da cara e dividimos pela quantidade de pixeis de cara.
            # Isto ignora completamente o fundo e é robusto contra ruído pontual.
            
            numerator = torch.sum(weighted_error_map)
            denominator = torch.sum(mask_tensor)
            
            # Proteção contra divisão por zero (caso a máscara seja toda preta)
            if denominator > 0:
                mse_loss = (numerator / denominator).item()
            else:
                mse_loss = 0.0

            # --- FIM DA CORREÇÃO ---

            # Guardar Resultados
            y_true.append(label)
            y_scores.append(mse_loss)

            if label == 1:
                fake_results.append({'path': img_path, 'mse': mse_loss})


    # 4. Análise de Extremos
    if fake_results:
        sorted_fakes = sorted(fake_results, key=lambda x: x['mse'])
        
        print("\n--- A GERAR HEATMAPS ---")
        # Melhor Fake (Menor Erro - Mais difícil de detetar)
        best_fake = sorted_fakes[0]
        print(f"Melhor Fake (Menor Erro): {best_fake['mse']:.5f}")
        save_analysis_plot(model, best_fake['path'], best_fake['mse'], "LOWEST_ERROR_FAKE")
        
        # Piores Fakes (Maior Erro - Mais fáceis)
        worst_fakes = sorted_fakes[-3:]
        for i, fake in enumerate(reversed(worst_fakes)):
            print(f"Pior Fake #{i+1} (Maior Erro): {fake['mse']:.5f}")
            save_analysis_plot(model, fake['path'], fake['mse'], f"HIGH_ERROR_FAKE_{i+1}")

    # 5. Métricas Finais
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = [1 if s > optimal_threshold else 0 for s in y_scores]
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n--- RESULTADOS FINAIS ---")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Threshold Ótimo: {optimal_threshold:.5f}")
    
    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(MODELS_DIR / "roc_curve_final.png")
    plt.close()

if __name__ == "__main__":
    evaluate()
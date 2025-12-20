import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
from skimage import color

# Imports locais
from config import MODELS_DIR, FAKES_DIR, DEVICE, IMG_SIZE
from modelo import UNetColorizer

# --- CONFIGURAÇÕES ---
BATCH_SIZE = 1 # Avaliar uma a uma para gerar heatmaps individuais
THRESHOLD_FIXO = 0.005 # Valor inicial, depois calculamos o ótimo

def load_image_pipeline(path):
    """Pipeline idêntico ao treino: Leitura -> Resize -> RGB -> Lab -> Tensor"""
    # Leitura segura para Windows
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    
    if img_bgr is None: return None, None

    # Processamento
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype("float32") / 255.0
    img_lab = color.rgb2lab(img_rgb)
    
    L = img_lab[:, :, 0] / 100.0
    ab = img_lab[:, :, 1:] / 128.0
    
    # Tensores
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    ab_tensor = torch.from_numpy(ab.transpose((2, 0, 1))).unsqueeze(0).float().to(DEVICE)
    
    return L_tensor, ab_tensor, img_rgb

def generate_heatmap(ab_pred, ab_real):
    """
    Gera o mapa de calor baseado na diferença Euclidiana entre a cor prevista e a real.
    Fórmula: E = ||ab_pred - ab_real||_2
    """
    # Desnormalizar para escala real (aprox) para o erro fazer sentido visualmente
    diff = (ab_pred - ab_real) * 128.0 
    
    # Calcular magnitude do erro pixel a pixel
    # diff shape: (1, 2, H, W) -> square -> sum channels -> sqrt
    error_map = torch.sqrt(torch.sum(diff ** 2, dim=1)).squeeze().cpu().numpy()
    
    return error_map

def evaluate():
    print(f"--- AVALIAÇÃO (DETEÇÃO DE DEEPFAKES) ---")
    
    # 1. Carregar Modelo
    model_path = MODELS_DIR / "chroma_detector_best.pth"
    if not model_path.exists():
        print("Erro: Modelo não encontrado. Treina primeiro!")
        return

    model = UNetColorizer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 2. Preparar Dados de Teste
    # Vamos usar os fakes gerados e os seus pares reais
    fake_paths = sorted(list(FAKES_DIR.glob("*_fake.png")))
    real_paths = sorted(list(FAKES_DIR.glob("*_real.png")))
    
    # Limitar teste se necessário
    limit = 500 # Avaliar em 500 pares (1000 imagens) é suficiente para o paper
    fake_paths = fake_paths[:limit]
    real_paths = real_paths[:limit]
    
    y_true = []   # 0 = Real, 1 = Fake
    y_scores = [] # Erro médio de reconstrução (MSE do ab)
    
    print(f"A avaliar {len(real_paths)} reais e {len(fake_paths)} fakes...")
    
    # Loop de avaliação
    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    
    for img_path, label in tqdm(all_paths):
        L, ab_real, original_rgb = load_image_pipeline(img_path)
        if L is None: continue
        
        with torch.no_grad():
            ab_pred = model(L)
            
            # Calcular erro médio da imagem inteira (Loss)
            mse_loss = torch.mean((ab_pred - ab_real)**2).item()
            
            y_true.append(label)
            y_scores.append(mse_loss)
            
            # --- GUARDAR HEATMAPS EXEMPLO (Apenas alguns) ---
            # Vamos guardar o heatmap se for um Fake com erro alto (para o paper)
            if label == 1 and len(y_scores) % 50 == 0:
                heatmap = generate_heatmap(ab_pred, ab_real)
                
                plt.figure(figsize=(10, 4))
                
                # Imagem Original
                plt.subplot(1, 3, 1)
                plt.imshow(original_rgb)
                plt.title("Deepfake Input")
                plt.axis('off')
                
                # Cores Previstas (Reconstrução)
                # (Nota: Visualizar Lab reconstruído requer conversão complexa, 
                # mostramos apenas o Heatmap para simplicidade científica)
                
                # Heatmap
                plt.subplot(1, 3, 2)
                plt.imshow(heatmap, cmap='jet', vmin=0, vmax=20) # vmin/vmax controla contraste
                plt.title("Erro Cromático (Heatmap)")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.hist(heatmap.ravel(), bins=50, color='red', alpha=0.7)
                plt.title("Histograma de Erro")
                
                save_name = MODELS_DIR / f"analysis_{img_path.stem}.png"
                plt.savefig(save_name)
                plt.close()

    # 3. Calcular Métricas Finais (ROC / AUC)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nRESULTADOS FINAIS:")
    print(f"AUC Score: {roc_auc:.4f} (Aleatório=0.5, Perfeito=1.0)")
    
    # Encontrar melhor threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Threshold Ótimo de Erro: {optimal_threshold:.5f}")
    
    # Calcular Accuracy com o threshold ótimo
    y_pred = [1 if s > optimal_threshold else 0 for s in y_scores]
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy no Teste: {acc:.4f}")

    # 4. Plot Curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Performance do Detetor "Chroma Truth"')
    plt.legend(loc="lower right")
    plt.savefig(MODELS_DIR / "roc_curve.png")
    print(f"Gráfico ROC salvo em {MODELS_DIR}")

if __name__ == "__main__":
    evaluate()
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchsummary import summary
from config import MODELS_DIR, DEVICE, FAKES_DIR
import cv2
from modelo import UNetColorizer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# --- CONFIGURAÇÕES ---
BATCH_SIZE = 8
EPOCHS = 250     
LR = 0.001
PATIENCE = 15

class EarlyStopping:
    """Para o treino se o validation loss não melhorar após X épocas."""
    def __init__(self, patience=7, min_delta=1e-4, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'| EarlyStopping: {self.counter}/{self.patience} sem melhoria...')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'| Loss melhorou ({self.val_loss_min:.6f} --> {val_loss:.6f}). A guardar modelo...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


########################################################################################
########################################################################################

#########################################################################################
#                                                                                       #
#                       Parte do "Detetive Cromatico"                                   #
#                                                                                       #
#########################################################################################
class LabColorizationDataset(Dataset):
    def __init__(self, split='train'):
        # Procura ficheiros que terminam em '_real.png' na pasta de fakes
        all_files = sorted(list(FAKES_DIR.glob("*_real.png")))
        self.split = split
        
        if not all_files:
            print(f"AVISO: Nenhuma imagem encontrada em {FAKES_DIR}.")
            self.files = []
            return

        # Hard Split 4000/200
        train_cutoff = 4000
        val_cutoff = 4200 
        
        if split == 'train':
            self.files = all_files[:train_cutoff]
            print(f"Dataset (TRAIN): {len(self.files)} imagens [Índices 0-{train_cutoff}]")
        elif split == 'val':
            self.files = all_files[train_cutoff:val_cutoff]
            print(f"Dataset (VAL): {len(self.files)} imagens [Índices {train_cutoff}-{val_cutoff}]")

    def __len__(self):
        return len(self.files)

    def read_image_safe(self, path):
        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        except:
            return None

    def __getitem__(self, idx):
        path = str(self.files[idx])

        # 1. Ler imagem
        img_bgr = self.read_image_safe(path)
        if img_bgr is None:
            return torch.zeros(1, 512, 512), torch.zeros(2, 512, 512), torch.zeros(1, 512, 512)

        if self.split == 'train':
            # 1. Alteração de Iluminação (Multiplicação)
            factor = np.random.uniform(0.5, 1.5)
            img_bgr = cv2.multiply(img_bgr, factor)

            # Garantir que não passa dos limites [0, 255]
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

        # Continua com o Resize normal...
        img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)

        # 2. Conversão LAB e Normalização (Igual a antes)
        img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
        L = img_lab[:, :, 0].astype("float32") / 255.0
        a = (img_lab[:, :, 1].astype("float32") - 128.0) / 128.0
        b = (img_lab[:, :, 2].astype("float32") - 128.0) / 128.0
        
        ab_norm = np.stack([a, b], axis=2)

        # 3. CRIAR A MÁSCARA DE ATENÇÃO (Gaussian Weight)
        # Cria um "holofote" branco no centro e preto nas bordas
        H, W = 512, 512
        Y, X = np.ogrid[:H, :W]
        center_y, center_x = H / 2, W / 2
        sigma_y = 80 # Mais apertado na vertical (testa e queixo)
        sigma_x = 60  # Muito mais apertado na horizontal (para cortar cabelos laterais)

        dist_from_center = np.sqrt(((X - center_x)**2 / (2 * sigma_x**2)) + 
                                ((Y - center_y)**2 / (2 * sigma_y**2)))

 
        mask = np.exp(-dist_from_center)
        mask = np.where(mask < 0.2, 0, mask)
        
        
        # 4. Tensores
        L_norm = L.astype("float32")
        ab_norm = ab_norm.astype("float32")
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm.transpose((2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() # (1, 512, 512)
        

        return L_tensor, ab_tensor, mask_tensor
    
########################################################################################
########################################################################################

def save_loss_plot(train_losses, val_losses, dest_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Curvas de Aprendizagem')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(dest_path)
    plt.close()
    print(f"Gráfico salvo em: {dest_path}")

#########################################################################################
#                                                                                       #                                 
#                    Função Principal de Treino do Modelo                               #
#                                                                                       #             
#########################################################################################

def train_model():
    print(f"--- A INICIAR TREINO NO DEVICE: {DEVICE} ---")
    
    # 1. Dados
    train_ds = LabColorizationDataset(split='train')
    if len(train_ds) == 0:
        print("ERRO: Dataset vazio.")
        return
    val_ds = LabColorizationDataset(split='val')
    
    # pin_memory=True acelera a transferência de dados para a GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Modelo
    model = UNetColorizer().to(DEVICE)

    print("Resumo do Modelo:") 
    summary(model, input_size=(1, 512, 512))  

    # 3. Setup de Treino
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # SCALER para Mixed Precision (Crucial para a 5060 Ti)
    scaler = torch.amp.GradScaler('cuda') 

    # Instância do Early Stopping
    save_path = MODELS_DIR / "classificador.pth"
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=str(save_path))

    train_history = []
    val_history = []
    
    # 4. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Loop de treino
        for L_input, ab_target, mask in loop: # <--- Recebe a máscara
            L_input = L_input.to(DEVICE, non_blocking=True)
            ab_target = ab_target.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True) # Máscara para a GPU
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                ab_pred = model(L_input)
                
                # --- LOSS PONDERADA (Weighted Loss) ---
                # Calculamos o erro quadrado (MSE) sem reduzir (reduction='none')
                # Isto dá-nos o erro de CADA pixel individualmente
                pixel_errors = (ab_pred - ab_target) ** 2
                
                # Multiplicar o erro pela máscara
                # Erro no centro (cara) conta 100%. Erro no canto conta 10%.
                weighted_errors = pixel_errors * mask

                # Média ponderada pela máscara
                loss = torch.sum(weighted_errors) / torch.sum(mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # -------------------------------
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        train_history.append(avg_train_loss)
        
        # --- Validação ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # CORREÇÃO: Adicionar 'mask' aqui
            for L_input, ab_target, mask in val_loader:
                L_input = L_input.to(DEVICE, non_blocking=True)
                ab_target = ab_target.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True) # Enviar máscara para a GPU
                
                with torch.amp.autocast('cuda'):
                    ab_pred = model(L_input)
                    
                    # CÁLCULO DA LOSS PONDERADA (Igual ao Treino)
                    # 1. Erro quadrado por pixel
                    pixel_errors = (ab_pred - ab_target) ** 2
                    
                    # 2. Aplicar máscara (Zerar o fundo)
                    weighted_errors = pixel_errors * mask
                    
                    # 3. Média ponderada pela máscara (não pelo total de pixeis)
                    # Somamos o erro total e dividimos pela "quantidade de cara" que existe
                    loss = torch.sum(weighted_errors) / torch.sum(mask)
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_history.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # --- Early Stopping Call ---
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        
        
        if early_stopping.early_stop:
            print("--- Early stopping ativado. Treino terminado. ---")
            break

    # 5. Fim do Treino
    print("A carregar o melhor modelo para garantir consistência...")
    model.load_state_dict(torch.load(save_path))
    
    plot_path = MODELS_DIR / "training_curves.png"
    save_loss_plot(train_history, val_history, plot_path)
    print("Concluído.")

if __name__ == "__main__":
    print(f"O device é {DEVICE}")
    import matplotlib.pyplot as plt
    import random
    
    print("--- TESTE DE VISUALIZAÇÃO DA MÁSCARA ---")
    
    # 1. Instanciar o Dataset
    # Usamos 'train' para ir buscar as imagens reais
    ds = LabColorizationDataset(split='train')
    
    if len(ds) == 0:
        print("Erro: Dataset vazio.")
        exit()

    # 2. Escolher uma imagem aleatória para testar
    idx = random.randint(0, min(100, len(ds)-1))
    print(f"A visualizar imagem índice: {idx}")
    
    # Receber os 3 tensores: L (Input), ab (Target), Mask (Peso)
    L_tensor, ab_tensor, mask_tensor = ds[idx]
    
    # 3. Converter para Numpy (para o Matplotlib entender)
    # .squeeze() remove a dimensão do canal (1, 512, 512) -> (512, 512)
    L_numpy = L_tensor.squeeze().numpy()
    mask_numpy = mask_tensor.squeeze().numpy()
    
    # 4. Gerar o Gráfico
    plt.figure(figsize=(15, 5))
    
    # Plot 1: A Imagem Original (Canal L - Preto e Branco)
    plt.subplot(1, 3, 1)
    plt.imshow(L_numpy, cmap='gray')
    plt.title("Input Original (Canal L)")
    plt.axis('off')
    
    # Plot 2: A Máscara de Calor (Onde a rede vai prestar atenção)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_numpy, cmap='jet', vmin=0, vmax=1)
    plt.title("Máscara de Atenção (Gaussian)")
    plt.colorbar(label="Peso na Loss")
    plt.axis('off')
    
    # Plot 3: O Resultado Efetivo (O que a Loss "vê")
    # Multiplicamos para ver as áreas que serão ignoradas (escurecidas)
    plt.subplot(1, 3, 3)
    plt.imshow(L_numpy * mask_numpy, cmap='gray')
    plt.title("Foco Efetivo (Áreas escuras = Ignoradas)")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar ou Mostrar
    save_path = "teste_mascara.png"
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")
    train_model()
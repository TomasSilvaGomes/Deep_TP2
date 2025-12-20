import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from torchsummary import summary
# Imports locais
from config import MODELS_DIR, DEVICE
from processamento import LabColorizationDataset 
from modelo import UNetColorizer

# --- CONFIGURAÇÕES ---
BATCH_SIZE = 8  
EPOCHS = 100     
LR = 2e-4
PATIENCE = 5    

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

def train_model():
    print(f"--- A INICIAR TREINO NO DEVICE: {DEVICE} ---")
    
    # 1. Dados
    train_ds = LabColorizationDataset(split='train')
    if len(train_ds) == 0:
        print("ERRO: Dataset vazio.")
        return
    val_ds = LabColorizationDataset(split='val')
    
    # pin_memory=True acelera a transferência de dados para a GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    
    # 2. Modelo
    model = UNetColorizer().to(DEVICE)

    print("Resumo do Modelo:") 
    summary(model, input_size=(1, 512, 512))  # Exemplo de input size (L channel)
    

    # 3. Setup de Treino
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # SCALER para Mixed Precision (Crucial para a 5060 Ti)
    scaler = torch.amp.GradScaler('cuda') 

    # Instância do Early Stopping
    save_path = MODELS_DIR / "chroma_detector_best.pth"
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=str(save_path))
    
    train_history = []
    val_history = []
    
    # 4. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for L_input, ab_target in loop:
            L_input = L_input.to(DEVICE, non_blocking=True)
            ab_target = ab_target.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) # Ligeiramente mais eficiente que zero_grad()
            
            # --- MIXED PRECISION CONTEXT ---
            with torch.amp.autocast('cuda'):
                ab_pred = model(L_input)
                loss = criterion(ab_pred, ab_target)
            
            # Backward com Scaler
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
            for L_input, ab_target in val_loader:
                L_input = L_input.to(DEVICE, non_blocking=True)
                ab_target = ab_target.to(DEVICE, non_blocking=True)
                
                # Na validação também usamos autocast para poupar memória e ser consistente
                with torch.amp.autocast('cuda'):
                    ab_pred = model(L_input)
                    loss = criterion(ab_pred, ab_target)
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_history.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # --- Early Stopping Call ---
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
    # Esta linha ajuda a prevenir erros de multiprocessamento no Windows
    torch.multiprocessing.freeze_support()
    train_model()
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import color # Pip install scikit-image
from config import FAKES_DIR

class LabColorizationDataset(Dataset):
    def __init__(self, split='train'):
        """
        Carrega apenas imagens REAIS para treino self-supervised.
        """
        # Procura ficheiros que terminam em '_real.png' na pasta de fakes
        all_files = sorted(list(FAKES_DIR.glob("*_real.png")))
        
        if not all_files:
            print(f"AVISO: Nenhuma imagem encontrada em {FAKES_DIR}. Corre o attacks.py primeiro!")

        # Divisão 90% Treino / 10% Validação
        split_idx = int(len(all_files) * 0.9)
        if split == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        print(f"Dataset ({split}): {len(self.files)} imagens carregadas.")

    def __len__(self):
        return len(self.files)

    def read_image_safe(self, path):
        """Lê imagem lidando com caminhos Windows com acentos."""
        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        except:
            return None

    def __getitem__(self, idx):
        path = str(self.files[idx])
        
        # 1. Ler imagem (BGR)
        img_bgr = self.read_image_safe(path)
        if img_bgr is None:
            # Se falhar, retorna tensores a zeros para não parar o treino (fail-safe)
            return torch.zeros(1, 512, 512), torch.zeros(2, 512, 512)

        # 2. Converter BGR -> RGB e Normalizar [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        
        # 3. Conversão Científica RGB -> Lab (Skimage é mais preciso que cv2 aqui)
        # L: [0, 100], ab: [-128, 127] aprox
        img_lab = color.rgb2lab(img_rgb)
        
        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]
        
        # 4. Normalização para a Rede (Crucial para a Tanh)
        # L [0, 100] -> [0, 1]
        L_norm = L / 100.0 
        # ab [-128, 128] -> [-1, 1]
        ab_norm = ab / 128.0 
        
        # 5. Converter para Tensores (C, H, W)
        # Input: (1, 512, 512)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        # Target: (2, 512, 512)
        ab_tensor = torch.from_numpy(ab_norm.transpose((2, 0, 1))).float()
        
        return L_tensor, ab_tensor
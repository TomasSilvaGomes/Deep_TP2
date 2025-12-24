import os
import torch
from pathlib import Path

# --- Deteção Automática de Caminhos ---
# Localização deste ficheiro (config.py)
FILE_PATH = Path(__file__).resolve()


PROJECT_ROOT = FILE_PATH.parent.parent

# --- Estrutura de Diretorias ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "originais"             # Onde estão os originais (1024x1024)
FAKES_DIR = DATA_DIR / "fakes"         # Onde ficam os pares processados
MODELS_DIR = PROJECT_ROOT / "save_models" # Pasta pedida para guardar os modelos

# --- Constantes Globais ---
IMG_SIZE = (512, 512) # Resolução alvo para Resize e SD
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Cria as pastas se não existirem
RAW_DIR.mkdir(parents=True, exist_ok=True)
FAKES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Debug para validares no terminal
print(f"--- CONFIG ---")
print(f"Raiz: {PROJECT_ROOT}")
print(f"Dados Raw: {RAW_DIR}")
print(f"Output Fakes: {FAKES_DIR}")
print(f"Modelos: {MODELS_DIR}")
print(f"Device: {DEVICE}")
print(f"--------------")
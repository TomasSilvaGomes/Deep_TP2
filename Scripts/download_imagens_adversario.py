import gdown
import zipfile
import os

# CONFIGURAÇÃO
FILE_ID = '1fP9pBRyy0yWC8ZqIJl6VcFXyaXn2FS0a' 
OUTPUT_ZIP = 'fakes.zip'
EXTRACT_ROOT = 'data' 
TARGET_DIR = os.path.join(EXTRACT_ROOT, 'fakes')

def setup_data():
    # 1. Validação Inteligente: Verifica se a pasta existe E se tem ficheiros
    # Se a pasta não existir OU se estiver vazia (lista de ficheiros vazia), faz download.
    if not os.path.exists(TARGET_DIR) or not os.listdir(TARGET_DIR):
        
        print(f"Dataset em falta ou vazio. A iniciar setup...")
        
        # Garante que a pasta raiz 'data' existe
        os.makedirs(EXTRACT_ROOT, exist_ok=True)
        
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        
        print(f"A baixar dataset ({OUTPUT_ZIP})...")
        gdown.download(url, OUTPUT_ZIP, quiet=False)
        
        print(f"A extrair para '{EXTRACT_ROOT}'...")
        with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_ROOT)
        
        os.remove(OUTPUT_ZIP)
        print(f"Sucesso! Imagens prontas em '{TARGET_DIR}'.")
        
    else:
        # Conta quantos ficheiros lá estão para confirmar
        count = len(os.listdir(TARGET_DIR))
        print(f"Dataset já verificado em '{TARGET_DIR}'. Contém {count} ficheiros.")

if __name__ == "__main__":
    setup_data()
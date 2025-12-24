import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Importar configurações do ficheiro config.py
from config import RAW_DIR, FAKES_DIR, IMG_SIZE, DEVICE

# --- FUNÇÕES AUXILIARES ---
def imread_unicode(path):
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None

def imwrite_unicode(path, img):
    try:
        is_success, im_buf_arr = cv2.imencode(".png", img)
        if is_success:
            with open(path, "wb") as f:
                im_buf_arr.tofile(f)
        return is_success
    except Exception as e:
        print(f"Erro ao gravar {path}: {e}")
        return False

# --- CLASSE DATASET (NOVO) ---
class DeepfakeDataset(Dataset):
    """
    Classe responsável por carregar imagens e prepará-las para o DataLoader.
    Isto permite que o CPU prepare o próximo batch enquanto a GPU trabalha.
    """
    def __init__(self, image_paths, img_size):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        img_cv = imread_unicode(path)
        
        # Se falhar a ler, devolvemos um marcador de erro
        if img_cv is None:
            return np.zeros((1,1,3), dtype=np.uint8), "ERROR"

        # Redimensionar e Converter para RGB
        img_resized = cv2.resize(img_cv, self.img_size, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # O DataLoader devolve Arrays/Tensores. O Pipeline do Diffusers espera PIL ou Tensores.
        # Vamos devolver numpy e converter para PIL no loop principal para flexibilidade.
        return img_rgb, path

# --- CLASSE DO GERADOR ---
class DeepfakeGenerator:
    def __init__(self):
        print(f"[Init] A carregar Stable Diffusion em: {DEVICE}...")
        
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=dtype,
                safety_checker=None 
            )
            self.pipe.to(DEVICE)
                            
        except Exception as e:
            print(f"ERRO CRÍTICO ao carregar modelo: {e}")
            exit()

    def create_batch_masks(self, images_pil):
        """Cria uma lista de máscaras para o batch atual."""
        masks = []
        for img in images_pil:
            w, h = img.size
            mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = int(h * 0.35), int(h * 0.65)
            x1, x2 = int(w * 0.25), int(w * 0.75)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            masks.append(Image.fromarray(mask))
        return masks

    def process_dataset(self, limit=None, batch_size=4):
        # 1. Listar Imagens
        all_images = sorted(list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg")))
        
        if not all_images:
            print(f"ERRO: Nenhuma imagem encontrada em {RAW_DIR}")
            return

        if limit:
            all_images = all_images[:limit]
            print(f"[Process] MODO TESTE: {limit} imagens.")
        else:
            print(f"[Process] A processar {len(all_images)} imagens.")

        # 2. Configurar DataLoader (A Magia do Batching)
        dataset = DeepfakeDataset(all_images, IMG_SIZE)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=False)
        
        print(f"[Info] Batch Size: {batch_size}. Total de Batches: {len(loader)}")
        print("Podes parar com Ctrl+C.")

        count = 0
        prompt = "clean face, realistic skin texture, perfect lighting, high quality"
        negative = "distortion, bad anatomy, blur, cartoon, makeup, artifacts, low quality"

        # 3. Loop de Batch
        try:
            for batch_imgs_np, batch_paths in loader:

                
                valid_imgs_pil = []
                valid_paths = []
                
                # Filtrar erros de leitura
                for i in range(len(batch_paths)):
                    if batch_paths[i] != "ERROR":
                        # Converter Tensor/Numpy -> PIL
                        img_array = batch_imgs_np[i].numpy().astype(np.uint8)
                        valid_imgs_pil.append(Image.fromarray(img_array))
                        valid_paths.append(batch_paths[i])
                
                if not valid_imgs_pil: continue

                # Criar máscaras
                batch_masks = self.create_batch_masks(valid_imgs_pil)

                # --- INFERÊNCIA (PARALELA NA GPU) ---
                with torch.no_grad():
                    # O pipeline aceita listas! Aqui acontece a magia do batch processing.
                    outputs = self.pipe(
                        prompt=[prompt] * len(valid_imgs_pil),
                        image=valid_imgs_pil,
                        mask_image=batch_masks,
                        negative_prompt=[negative] * len(valid_imgs_pil),
                        num_inference_steps=20,
                        guidance_scale=7.5
                    ).images

                # --- GUARDAR RESULTADOS ---
                for original_pil, fake_pil, path_str in zip(valid_imgs_pil, outputs, valid_paths):
                    p = Path(path_str)
                    base_name = p.stem
                    
                    # PIL -> BGR (OpenCV)
                    real_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                    fake_cv = cv2.cvtColor(np.array(fake_pil), cv2.COLOR_RGB2BGR)
                    
                    imwrite_unicode(str(FAKES_DIR / f"{base_name}_real.png"), real_cv)
                    imwrite_unicode(str(FAKES_DIR / f"{base_name}_fake.png"), fake_cv)
                    count += 1
                
                print(f"Progresso: {count} imagens processadas...", end='\r')

        except KeyboardInterrupt:
            print("\nParado pelo utilizador.")
        except Exception as e:
            print(f"\nErro inesperado no loop: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nConcluído! {count} pares gerados na pasta: {FAKES_DIR}")

if __name__ == "__main__":
    # Necessário para num_workers > 0 no Windows
    torch.multiprocessing.freeze_support()
    
    generator = DeepfakeGenerator()
    
    generator.process_dataset(limit=5000, batch_size=4)
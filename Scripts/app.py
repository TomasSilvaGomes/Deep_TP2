import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

# Importar configurações e modelo do teu projeto
from config import DEVICE, MODELS_DIR, HEATMAPS_DIR
from modelo import UNetColorizer

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Chroma Truth | Detetor de Deepfakes",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÕES AUXILIARES ---

@st.cache_resource
def load_model():
    """Carrega o modelo apenas uma vez (Cache) para ser rápido."""
    model_path = MODELS_DIR / "classificador.pth"
    if not model_path.exists():
        return None
    
    model = UNetColorizer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def get_attention_mask():
    """Gera a máscara gaussiana exata usada no treino."""
    H, W = 512, 512
    Y, X = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    # Sigma igual ao treino/avaliação
    sigma_y = 100 
    sigma_x = 80 
    dist_from_center = np.sqrt(((X - center_x)**2 / (2 * sigma_x**2)) + 
                            ((Y - center_y)**2 / (2 * sigma_y**2)))
    mask = np.exp(-dist_from_center)
    mask = np.where(mask < 0.2, 0, mask)
    mask = np.maximum(mask, 0.0)
    return mask

def process_image_in_memory(uploaded_file):
    """
    Processa a imagem diretamente da memória sem guardar no disco.
    Evita o erro [WinError 32] no Windows.
    """
    # 1. Ler bytes do ficheiro carregado
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return None, None, None

    # 2. Resize
    img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)

    # 3. Conversão BGR -> Lab
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
    
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    # 4. Normalização
    L_norm = L.astype("float32") / 255.0
    a_norm = (a.astype("float32") - 128.0) / 128.0
    b_norm = (b.astype("float32") - 128.0) / 128.0
    
    ab_norm = np.stack([a_norm, b_norm], axis=2)
    
    # 5. Tensores
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    # 6. RGB Visual
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype("float32") / 255.0
    
    return L_tensor, ab_norm, img_rgb # Retorna ab_norm numpy para calculo erro

def process_upload(uploaded_file, model, threshold):
    """Lógica de Inferência adaptada para Streamlit (RAM Only)."""
    
    # Pipeline Direta da Memória
    L, ab_real_np, original_rgb = process_image_in_memory(uploaded_file)
    
    if L is None:
        st.error("Erro ao ler a imagem.")
        return

    # Converter ab_real numpy para tensor GPU para cálculo
    ab_real = torch.from_numpy(ab_real_np.transpose((2, 0, 1))).unsqueeze(0).float().to(DEVICE)

    # 3. Predição
    with torch.no_grad():
        ab_pred = model(L)

    # 4. Máscara
    mask_np = get_attention_mask()
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    mask_visual = mask_np[:, :, np.newaxis]

    # 5. Cálculo do Erro (Métrica Global)
    squared_diff = (ab_pred - ab_real)**2
    weighted_error_map = squared_diff * mask_tensor
    
    numerator = torch.sum(weighted_error_map)
    denominator = torch.sum(mask_tensor)
    
    mse_loss = (numerator / denominator).item() if denominator > 0 else 0.0

    # 6. Preparar Visualização
    diff = (ab_pred - ab_real) * 128.0 
    heatmap_tensor = torch.sqrt(torch.sum(diff ** 2, dim=1)).squeeze().cpu().numpy()
    heatmap_masked = heatmap_tensor * mask_np
    
    # Aplicar máscara no input para mostrar o foco
    input_masked = original_rgb * mask_visual

    # --- EXIBIÇÃO ---
    st.divider()
    
    # Colunas de Métricas
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric("Erro Cromático (MSE)", f"{mse_loss:.5f}")
    
    with col_res2:
        st.metric("Threshold Atual", f"{threshold:.5f}")

    with col_res3:
        if mse_loss > threshold:
            st.error("🚨 RESULTADO: FAKE")
        else:
            st.success("✅ RESULTADO: REAL")

    # Colunas de Imagens
    st.subheader("Análise Visual (XAI)")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # CORREÇÃO DO AVISO: width="stretch" em vez de use_container_width=True
        st.image(input_masked, caption="Foco da Rede (Máscara Aplicada)", width="stretch")
    
    with col2:
        # Heatmap com Matplotlib
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap_masked, cmap='jet', vmin=0, vmax=30)
        plt.colorbar(im)
        plt.axis('off')
        st.pyplot(fig)
        st.caption("Mapa de Calor do Erro")

    with col3:
        # Histograma
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(heatmap_masked[mask_np > 0], bins=50, color='red', alpha=0.7, range=(0, 60))
        ax_hist.set_title("Distribuição do Erro")
        st.pyplot(fig_hist)
        st.caption("Histograma de Anomalias")


# --- INTERFACE PRINCIPAL ---

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/UBI_Logo.png", width=150) # Logo Opcional
st.sidebar.title("Chroma Truth")
mode = st.sidebar.radio("Navegação", ["Detetor (Demo)", "Relatório de Avaliação", "Sobre"])

if mode == "Detetor (Demo)":
    st.title("🕵️ Detetive Cromático")
    st.markdown("""
    Este sistema analisa a coerência física entre a **Luminância** (Luz) e a **Crominância** (Cor) de uma face.
    Deepfakes gerados por difusão podem falhar nesta correlação em condições complexas.
    """)

    model = load_model()
    
    if model is None:
        st.warning("⚠️ Modelo não encontrado! Corre o `autoencoder.py` primeiro para treinar.")
    else:
        # Sidebar Controls
        st.sidebar.header("Parâmetros")
        threshold = st.sidebar.slider("Sensibilidade (Threshold)", 0.001, 0.100, 0.008, format="%.4f")
        
        uploaded_file = st.file_uploader("Carregar imagem facial (Real ou Fake)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            if st.button("Analisar Imagem", type="primary"):
                with st.spinner('A processar a consistência cromática...'):
                    process_upload(uploaded_file, model, threshold)

elif mode == "Relatório de Avaliação":
    st.title("📊 Métricas do Modelo")
    
    # Verificar se existem imagens salvas pelo evaluate()
    roc_path = MODELS_DIR / "roc_curve_final.png"
    
    if roc_path.exists():
        st.image(str(roc_path), caption="Curva ROC Final", width="stretch")
        st.info("Para recalcular estas métricas, corre o script `detecao_XAI.py` no terminal.")
    else:
        st.warning("Ainda não foi gerado o gráfico ROC. Corre a Opção 2 no `main.py` primeiro.")

    # Mostrar Heatmaps de exemplo se existirem
    st.subheader("Exemplos do Dataset de Teste")
    
    # Procura imagens na pasta de heatmaps
    if HEATMAPS_DIR.exists():
        images = list(HEATMAPS_DIR.glob("*.png"))
        if images:
            selected_img = st.selectbox("Escolher análise guardada:", [i.name for i in images])
            st.image(str(HEATMAPS_DIR / selected_img), caption=selected_img, width="stretch")
        else:
            st.text("Nenhum heatmap guardado encontrado.")

elif mode == "Sobre":
    st.title("Projeto Final de Investigação")
    st.markdown("""
    **Universidade da Beira Interior** *Inteligência Artificial e Ciência de Dados - 2025/2026*
    
    **Autor:** Tomás Gomes
    
    **Metodologia:**
    1. **Self-Supervised Learning:** Treino de uma U-Net apenas em imagens reais.
    2. **Hypothesis:** Manipulações digitais quebram a coerência L-ab.
    3. **Attention Masking:** Foco gaussiano na região facial para ignorar ruído de fundo.
    """)
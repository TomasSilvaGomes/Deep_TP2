import sys
import os

# Importar as funções dos outros scripts existentes
from download_imagens_adversario import setup_data
from autoencoder import train_model
from detecao_XAI import evaluate

def run_full_pipeline():
    print("=======================================================")
    print("   CHROMA TRUTH | PIPELINE DE EXECUÇÃO TOTAL")
    print("=======================================================")
    
    # 1. Preparação dos Dados (Usa o teu script existente)
    print("\n[FASE 1/3] VERIFICAÇÃO E DOWNLOAD DE DADOS")
    try:
        setup_data()
    except Exception as e:
        print(f"Erro crítico no download/extração: {e}")
        sys.exit(1)
    
    # 2. Treino
    print("\n[FASE 2/3] TREINO DO AUTOENCODER (SELF-SUPERVISED)")
    # O train_model já tem as suas próprias impressões e barras de progresso
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTreino interrompido pelo utilizador. A avançar para avaliação...")
    
    # 3. Avaliação
    print("\n[FASE 3/3] AVALIAÇÃO E GERAÇÃO DE RELATÓRIOS")
    evaluate()
    
    print("\n=======================================================")
    print("   PIPELINE CONCLUÍDA.")
    print("=======================================================")

if __name__ == "__main__":
    run_full_pipeline()
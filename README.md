# Chroma Truth: Deteção de Deepfakes via Reconstrução Cromática 

**Trabalho de Investigação** **REDES NEURONAIS E APRENDIZAGEM PROFUNDA** **Universidade da Beira Interior - 2025/2026**

##  Sobre o Projeto

O **Chroma Truth** investiga uma nova abordagem para a deteção de Deepfakes baseada na física da cor. A hipótese central é que, enquanto a iluminação (Canal L) e a cor (Canais ab) de uma face real possuem uma coerência física intrínseca, as manipulações digitais (Deepfakes) quebram subtilmente esta relação.

Utilizamos uma **U-Net** treinada em regime **Self-Supervised** apenas com imagens reais. O modelo aprende a colorir faces; se falhar ao tentar colorir uma imagem de teste (gerando um erro alto), assume-se que a imagem é falsa ou contém anomalias de iluminação.

---

##  Estrutura do Projeto (Pipeline)

O projeto está modularizado na pasta `Scripts/` para garantir reprodutibilidade e organização:

* **`main.py`**   
  * Executa o fluxo completo: verifica/baixa o dataset, treina o modelo e gera os relatórios finais automaticamente.
* **`download_imagens_adversario.py`**   
  * Script auxiliar que descarrega o dataset da Google Drive se este não existir localmente.
* **`autoencoder.py`**   
  * Contém a lógica de treino da U-Net, Data Augmentation e definição da Loss Function.
* **`detecao_XAI.py`**   
  * Script de avaliação. Gera as curvas ROC/AUC, calcula o threshold ótimo e cria os Heatmaps (mapas de calor) para explicar onde o modelo focou.
* **`app.py`**   
  * Interface Web interativa (Dashboard) criada em Streamlit para demonstração em tempo real.
* **`modelo.py`** & **`config.py`**   
  * Definições da arquitetura da rede e variáveis globais (caminhos, hiperparâmetros).

---

##  Como Executar

### 1. Instalação das Dependências

Certifica-te de que tens o Python (3.9+) instalado. Instala as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```


## 2. Reprodução Total (Treino + Avaliação)
Para reproduzir todos os resultados do zero (download de dados, treino e geração de gráficos), corre apenas este comando na pasta raiz do projeto:

```Bash

python Scripts/main.py
```
O script irá verificar se os dados existem. Se não, fará o download automático, seguido do treino e da avaliação.

## 3. Demo Interativa (Streamlit)
Para utilizar o Dashboard visual ("Detetive Cromático"), é obrigatório entrar na pasta dos scripts antes de iniciar a aplicação para garantir que os caminhos relativos funcionam corretamente.

Passo a passo:

### Muda para a diretoria dos scripts:
```Bash
cd Scripts
```
### Inicia a aplicação:
```Bash
streamlit run app.py
```

# Resultados Esperados
Após a execução da pipeline, os seguintes artefactos serão gerados nas pastas Models/ e Heatmaps/:

* **classificador.pth**: Os pesos do modelo treinado.

* **roc_curve_final.png**: A curva de desempenho do classificador (AUC/ROC).

* **training_curves.png**: Gráfico da evolução da Loss durante o treino.

* **_fake.png**: Exemplos de imagens analisadas (Explicabilidade/XAI) com a máscara de atenção e o mapa de calor do erro aplicado.

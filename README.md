# Ararat-Model: Pipeline de Radiômica para Análise Oncológica

O Ararat-Model é um pipeline de software de ponta a ponta projetado para realizar análises de imagens médicas no campo da oncologia através da Radiômica. Seu objetivo principal é transformar dados visuais brutos (imagens de ressonância magnética) em um conjunto de dados quantitativos e estruturados, aptos para o treinamento de modelos avançados e interpretáveis de Inteligência Artificial (IA).

Atualmente, o projeto foca na classificação de lesões de câncer de próstata utilizando o banco de dados público PROSTATEx, visando prever o Grau de Gleason, um indicador crucial da agressividade do tumor. A filosofia central do projeto é a automação, reprodutibilidade e explicabilidade (XAI).

## Estrutura do Projeto

O projeto é organizado de forma modular para garantir clareza e facilidade de manutenção:

-   **/classes/**: Contém a lógica principal do pipeline de extração de features (`pipeline.py`, `segmentation.py`).
-   **/data/**: Repositório para os dados brutos (imagens DICOM, planilhas). *Ignorado pelo Git.*
-   **/results/**: Pasta de destino para todos os gráficos e visualizações gerados pela análise.
-   `main.py`: Ponto de entrada para a extração de features (Fase 1).
-   `analyze_shap.py`: Ponto de entrada para o treinamento do modelo, análise e interpretação (Fase 2).
-   `config.yaml`: Arquivo central para configurar todos os parâmetros do projeto.
-   `requirements.txt`: Lista de todas as dependências Python do projeto.

---

## Guia Rápido: Como Rodar o Pipeline Completo

Para executar o fluxo completo do Ararat-Model, siga os passos abaixo.

### 1. Pré-requisitos e Configuração Inicial

Antes de tudo, certifique-se de que o ambiente está corretamente configurado.

#### a. Estrutura de Dados
O projeto espera que a pasta `/data` contenha os seguintes arquivos (ela é ignorada pelo Git, então você precisa adicioná-los manualmente):

ARARAT-MODEL-MAIN/
└── data/
├── PROSTATEx/
│ ├── ProstateX-0000/
│ │ ├── ... (séries de imagens DICOM)
│ │ └── segmentation_ProstateX-0000_finding-1.nii.gz <- Exemplo de máscara
│ ├── ProstateX-0001/
│ └── ...
└── ProstateX-2-Findings-Train.csv

> **Nota:** É crucial que as máscaras de segmentação estejam presentes para cada lesão que se deseja analisar. A convenção de nomenclatura das máscaras deve ser ajustada no arquivo `classes/segmentation.py`.

#### b. Instalação das Dependências
Crie um ambiente virtual e instale todas as bibliotecas necessárias.

```bash
# 1. Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # No Linux/macOS
# .\.venv\Scripts\activate    # No Windows (PowerShell)

# 2. Instale todas as dependências listadas
pip install -r requirements.txt

. Configuração do Pipeline (config.yaml)
O arquivo config.yaml controla todo o comportamento do projeto. Antes de executar, você pode revisar e ajustar parâmetros como:
Quais sequências de RM analisar (extraction_tasks).
Se deve aplicar pré-processamento (preprocessing).
Se deve aplicar seleção de features e quantas manter (analysis_settings).
2. Fase 1: Extração de Features Radiômicas
Este passo lê as imagens DICOM, aplica as máscaras de segmentação e extrai mais de 100 features radiômicas para cada lesão, salvando-as em arquivos CSV.
Para executar a Fase 1, rode o seguinte comando no terminal:

python main.py
Use code with caution.

Ao final, você terá os arquivos radiomics_features_t2.csv e radiomics_features_adc.csv (ou outros, conforme configurado) na raiz do projeto.
3. Fase 2: Treinamento, Análise e Interpretabilidade
Este passo carrega os CSVs gerados, combina os dados, treina múltiplos modelos de machine learning usando AutoML, e realiza uma análise de interpretabilidade profunda com SHAP.
Para executar a Fase 2, rode o seguinte comando no terminal:

python analyze_shap.py
Use code with caution.


4. Análise dos Resultados
Após a conclusão da Fase 2, todos os resultados estarão disponíveis:
Modelos Treinados: Dentro da pasta AutoML_SHAP_Analysis/, você encontrará uma pasta para cada modelo treinado, contendo os artefatos do modelo, o leaderboard de performance, etc.
Gráficos de Interpretabilidade: Na pasta /results, você encontrará todos os gráficos SHAP gerados, como:
shap_feature_importance.png: Importância geral das features.
shap_summary_plot_geral.png: Impacto e direção de cada feature.
shap_dependence_plot_...: Relação entre uma feature e seu impacto.
Relatório de Performance no Terminal: Ao final da execução, um relatório detalhado com a acurácia, AUC, precision e recall (calculados com validação cruzada por paciente) será exibido diretamente no terminal.
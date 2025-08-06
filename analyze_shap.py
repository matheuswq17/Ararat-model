import pandas as pd
import shap
import numpy as np
from supervised.automl import AutoML
import matplotlib.pyplot as plt
import os
import shutil


from supervised.algorithms.lightgbm import LightgbmAlgorithm
from supervised.algorithms.xgboost import XgbAlgorithm
from supervised.algorithms.random_forest import RandomForestAlgorithm

# define os nomes das pastas de resultado
RESULTS_PATH = "AutoML_SHAP_Analysis"
RESULTS_FOLDER_PLOTS = "results"

# pasta pra salvar os graficos, se ela nao existir
if not os.path.exists(RESULTS_FOLDER_PLOTS):
    os.makedirs(RESULTS_FOLDER_PLOTS)

# limpa os resultados antigos pra comecar um treino do zero
if os.path.exists(RESULTS_PATH):
    print(f"Deletando a pasta de resultados antiga: {RESULTS_PATH}")
    shutil.rmtree(RESULTS_PATH)

# carrega os Dados
print("Carregando o arquivo de features...")
try:
    df = pd.read_csv("radiomics_features_final.csv")
except FileNotFoundError:
    print("\nERRO CRÍTICO: O arquivo 'radiomics_features_final.csv' não foi encontrado.")
    print("Por favor, execute 'python main.py' primeiro para gerar este arquivo.")
    exit()

TARGET_COLUMN = 'ggg'
COLS_TO_DROP = [
    'PatientID', 'FindingID', 'ggg', 'zone', 'ClinSig'
] + [col for col in df.columns if 'diagnostics_' in col]
X = df.drop(columns=COLS_TO_DROP, errors='ignore')
y = df[TARGET_COLUMN]
print(f"Dados carregados. Shape de X: {X.shape}, Shape de y: {y.shape}")

# treinar o modelo
print(f"Iniciando um novo treinamento para a pasta: {RESULTS_PATH}")
automl = AutoML(
    results_path=RESULTS_PATH,
    mode="Explain",
    algorithms=["Random Forest", "Xgboost", "LightGBM"],
    n_jobs=-1,
    total_time_limit=300
)
automl.fit(X, y)
print("Treinamento finalizado.")

# CARREGA O MODELO CAMPEAO DO DISCO E CALCULAR SHAP
print("Identificando e carregando o modelo campeão do disco...")
leaderboard_path = os.path.join(RESULTS_PATH, "leaderboard.csv")
leaderboard = pd.read_csv(leaderboard_path)
best_individual_model_name = leaderboard[leaderboard['model_type'] != 'Ensemble'].iloc[0]['name']
best_model_type = leaderboard[leaderboard['name'] == best_individual_model_name]['model_type'].iloc[0]
print(f"O melhor modelo individual identificado é: '{best_individual_model_name}' do tipo '{best_model_type}'")
specific_model_path_dir = os.path.join(RESULTS_PATH, best_individual_model_name)

model_file_name = None
expected_extensions = (".lightgbm", ".xgboost", ".random_forest", ".joblib")
for f in os.listdir(specific_model_path_dir):
    if f.endswith(expected_extensions):
        model_file_name = f
        break
if model_file_name is None:
    raise RuntimeError(f"Não foi possível encontrar o arquivo do modelo com uma das extensões {expected_extensions} na pasta '{specific_model_path_dir}'")
full_model_file_path = os.path.join(specific_model_path_dir, model_file_name)
print(f"Arquivo do modelo encontrado em: {full_model_file_path}")

model_framework = None
if best_model_type == 'LightGBM': model_framework = LightgbmAlgorithm({})
elif best_model_type == 'Xgboost': model_framework = XgbAlgorithm({})
elif best_model_type == 'Random Forest': model_framework = RandomForestAlgorithm({})
else: raise ValueError(f"Tipo de modelo '{best_model_type}' não suportado.")

model_framework.load(full_model_file_path)
model_to_explain_obj = model_framework.model

print("Modelo carregado com sucesso, iniciando cálculo dos valores SHAP...")
explainer = shap.TreeExplainer(model_to_explain_obj)
shap_values = explainer.shap_values(X)
print("Valores SHAP calculados com sucesso.")

#gerar os graficos
print("\nGerando e salvando os gráficos SHAP...")

#grafico de importancia (Bar Plot)
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
plt.title("Importância das Features (Média do Impacto Absoluto SHAP)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, "shap_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  - Grafico 'shap_feature_importance.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")

# grafico de resumo (Beeswarm Plots, abelha e relampago)
#verifica se o shap_values é uma matriz 3D
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    print("Detectado output multiclasse. Gerando um gráfico de resumo por classe.")
    n_classes = shap_values.shape[2]
    # usa os nomes de classe do target 'y' se possível senão apenas numeros
    class_names = sorted(y.unique()) if len(sorted(y.unique())) == n_classes else range(n_classes)

    for i, class_name in enumerate(class_names):
        plt.figure()
        shap.summary_plot(shap_values[:, :, i], X, show=False, max_display=20)
        plt.title(f"Impacto das Features na Previsão da Classe {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, f"shap_summary_plot_class_{class_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Grafico 'shap_summary_plot_class_{class_name}.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")
else:
    # fallback para o caso de classificacao binaria ou regressao
    print("Detectado output de classe única/binária. Gerando um gráfico de resumo geral.")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title("Impacto das Features na Previsão")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, "shap_summary_plot_geral.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Gráfico 'shap_summary_plot_geral.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")

# graficos adicionais
print("\nGerando gráficos adicionais (Dependence e Decision Plots)...")

# gerar Dependence Plots
# vamos pegar as 3 features mais importantes do grafico de barras

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
if mean_abs_shap.ndim > 1: # Caso multiclasse
    mean_abs_shap = np.mean(mean_abs_shap, axis=1)

shap_importance_df = pd.DataFrame({'feature': X.columns, 'importance': mean_abs_shap}).sort_values('importance', ascending=False)
top_3_features = shap_importance_df['feature'].head(3).tolist()
print(f"As 3 features mais importantes para os Dependence Plots são: {top_3_features}")

for feature_name in top_3_features:
    # o Dependence Plot mostra o efeito de uma feature, colorido por outra para ver interacoes
    shap.dependence_plot(
        feature_name, shap_values if shap_values.ndim < 3 else shap_values[:,:,-1], X,
        display_features=X, interaction_index="auto", show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, f"shap_dependence_plot_{feature_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Gráfico 'shap_dependence_plot_{feature_name}.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")

# gerar Decision Plots
predicted_probas = model_to_explain_obj.predict(X)
proba_high_grade = predicted_probas[:, -1]
try:
    low_proba_idx = np.where(proba_high_grade < 0.2)[0][0]
    mid_proba_idx = np.where((proba_high_grade > 0.3) & (proba_high_grade < 0.7))[0][0]
    high_proba_idx = np.where(proba_high_grade > 0.8)[0][0]
    example_indices = { "low_grade_example": low_proba_idx, "mid_grade_example": mid_proba_idx, "high_grade_example": high_proba_idx }
    print(f"Índices dos exemplos para os Decision Plots: {example_indices}")

    base_value_for_plot = explainer.expected_value[-1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap_values_for_plot = shap_values[-1] if isinstance(shap_values, list) else shap_values[:,:,-1]
    for name, idx in example_indices.items():
        shap.decision_plot(
            base_value_for_plot,
            shap_values_for_plot[idx],
            X.iloc[idx],
            feature_names=X.columns.tolist(),
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, f"shap_decision_plot_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Gráfico 'shap_decision_plot_{name}.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")
except IndexError:
    print("  - Não foi possível encontrar exemplos para todos os níveis de probabilidade para os Decision Plots. Pulando esta etapa.")

print("\nAnálise SHAP concluída com sucesso! Verifique os arquivos .png na pasta do projeto.")
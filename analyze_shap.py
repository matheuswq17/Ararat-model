import pandas as pd
import shap
import numpy as np
from supervised.automl import AutoML
import matplotlib.pyplot as plt
import os
import shutil

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score, classification_report
from supervised.automl import AutoML 

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
print("Carregando e combinando arquivos de features (T2 e ADC)...")
try:
    df_t2 = pd.read_csv("radiomics_features_t2.csv")
    df_adc = pd.read_csv("radiomics_features_adc.csv")
    cols_to_keep_adc = ['PatientID', 'FindingID'] + [col for col in df_adc.columns if 'original_' in col or 'wavelet_' in col]
    df_adc_features = df_adc[cols_to_keep_adc]
    df = pd.merge(df_t2, df_adc_features, on=['PatientID', 'FindingID'], suffixes=('_t2', '_adc'))
    print("Features combinadas com sucesso!")
except FileNotFoundError as e:
    print(f"\nERRO CRÍTICO: Não foi possível encontrar um dos arquivos de features: {e.filename}")
    print("Certifique-se de ter gerado tanto 'radiomics_features_t2.csv' quanto 'radiomics_features_adc.csv'.")
    exit()

groups = df['PatientID']
TARGET_COLUMN = 'ggg'
COLS_TO_DROP = [
    'PatientID', 'FindingID', 'ggg', 'zone', 'ClinSig'
] + [col for col in df.columns if 'diagnostics_' in col or '_adc' in col and col not in cols_to_keep_adc]
X = df.drop(columns=COLS_TO_DROP, errors='ignore')
y = df[TARGET_COLUMN]
print(f"Dados carregados. Shape de X: {X.shape}, Shape de y: {y.shape}")

y = y.apply(lambda ggg: 1 if ggg >= 4 else 0)
print("Problema transformado em classificação binária (0: Baixo/Médio Risco, 1: Alto Risco).")
print(f"Distribuição das classes:\n{y.value_counts(normalize=True)}")

APPLY_FEATURE_SELECTION = True
N_FEATURES_TO_SELECT = 40

if APPLY_FEATURE_SELECTION:
    print(f"\nIniciando Seleção de Features para manter as top {N_FEATURES_TO_SELECT} features...")
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=0.1, verbose=0)
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.support_]
    X = X[selected_features]
    
    print(f"Seleção concluída. Novo shape de X: {X.shape}")

# treinar o modelo
print(f"\nIniciando um novo treinamento para a pasta: {RESULTS_PATH}")

automl = AutoML(
    results_path=RESULTS_PATH,
    mode="Explain",
    eval_metric='auc',
    algorithms=["Random Forest", "Xgboost", "LightGBM"],
    n_jobs=-1,
    total_time_limit=600
)

group_kfold = GroupKFold(n_splits=5)
folds = list(group_kfold.split(X, y, groups))
automl.fit(X, y, cv=folds)

print("Treinamento finalizado.")

# CARREGA O MODELO CAMPEAO DO DISCO E CALCULAR SHAP
print("Identificando o modelo campeão do disco...")
leaderboard_path = os.path.join(RESULTS_PATH, "leaderboard.csv")
leaderboard = pd.read_csv(leaderboard_path)

best_individual_model_info = leaderboard[leaderboard['model_type'] != 'Ensemble'].iloc[0]
best_model_name = best_individual_model_info['name']
best_model_type = best_individual_model_info['model_type']

print(f"O melhor modelo INDIVIDUAL identificado para SHAP é: '{best_model_name}' do tipo '{best_model_type}'")

if best_model_type == 'LightGBM':
    model_loader = LightgbmAlgorithm({})
elif best_model_type == 'Xgboost':
    model_loader = XgbAlgorithm({})
elif best_model_type == 'Random Forest':
    model_loader = RandomForestAlgorithm({})
else:
    raise ValueError(f"Tipo de modelo '{best_model_type}' não suportado para carregamento manual.")

model_file_path = os.path.join(RESULTS_PATH, best_model_name, "learner_fold_0.joblib")
if not os.path.exists(model_file_path):
    model_file_path = os.path.join(RESULTS_PATH, best_model_name, "learner_fold_0.lightgbm")

model_loader.load(model_file_path)
model_to_explain_obj = model_loader.model

print("Modelo carregado com sucesso, iniciando cálculo dos valores SHAP...")
explainer = shap.Explainer(model_to_explain_obj, X)

shap_values = explainer(X, check_additivity=False)

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
mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({'feature': X.columns, 'importance': mean_abs_shap_values}).sort_values('importance', ascending=False)
top_5_features = shap_importance_df['feature'].head(5).tolist()
print(f"As 5 features mais importantes para os Dependence Plots são: {top_5_features}")

for feature_name in top_5_features:
    shap.dependence_plot(
        feature_name, shap_values.values, X,
        display_features=X, interaction_index="auto", show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, f"shap_dependence_plot_{feature_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Gráfico 'shap_dependence_plot_{feature_name}.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")

# gerar Decision Plots
try:
    base_value_for_plot = shap_values.base_values[0]
    shap_values_for_plots = shap_values.values

    raw_scores = model_to_explain_obj.predict(X)
    predicted_probas = 1 / (1 + np.exp(-raw_scores))
    
    low_proba_idx = np.where(predicted_probas < 0.2)[0][0]
    mid_proba_idx = np.where((predicted_probas > 0.4) & (predicted_probas < 0.6))[0][0]
    high_proba_idx = np.where(predicted_probas > 0.8)[0][0]
    example_indices = { "low_grade_example": low_proba_idx, "mid_grade_example": mid_proba_idx, "high_grade_example": high_proba_idx }
    print(f"Índices dos exemplos para os Decision Plots: {example_indices}")

    for name, idx in example_indices.items():
        shap.decision_plot(
            base_value_for_plot, shap_values_for_plots[idx], X.iloc[idx],
            feature_names=X.columns.tolist(), show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FOLDER_PLOTS, f"shap_decision_plot_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Gráfico 'shap_decision_plot_{name}.png' salvo em '{RESULTS_FOLDER_PLOTS}'.")
except IndexError:
    print("  - Nao foi possivel encontrar exemplos para todos os niveis de probabilidade para os Decision Plots. Pulando esta etapa.")

# imprimir resumo da performance com validacao cruzada
print("\n" + "="*50)
print("     RESUMO DA PERFORMANCE DO MELHOR MODELO (com Validação Cruzada por Paciente)")
print("="*50)

# usamos a informaçao do melhor modelo INDIVIDUAL, para consistência com a analise shap
best_model_info = best_individual_model_info 
print(f"Melhor Modelo para Análise: {best_model_info['name']}")
print(f"AUC Médio (Validação Cruzada Nativa do MLJAR): {best_model_info['metric_value']:.4f}")

# para obter um relatório de classificaçao detalhado, precisamos das predicoes 'out-of-fold'
# vamos gerar essas predicoes usando um estimador virgem do mesmo tipo do nosso melhor modelo
print("\nGerando predições 'out-of-fold' para o relatório de classificação...")

# importamos a funcao e os estimadores scikit-learn necessarios
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier # ja importado, mas bom ter aqui

# criamos um estimador virgem (nao treinado) do tipo correto
# o cross_val_predict vai clonar e treinar isso em cada fold
if best_model_type == 'LightGBM':
    # usamos os parametros padrao para maior precisão, poderiamos extrair os 
    # hiperparametros do `params.json` na pasta do modelo.
    final_estimator = LGBMClassifier(random_state=42) 
elif best_model_type == 'Xgboost':
    final_estimator = XGBClassifier(random_state=42)
elif best_model_type == 'Random Forest':
    final_estimator = RandomForestClassifier(random_state=42)

y_pred_cv = cross_val_predict(
    final_estimator, # passamos o estimador virgem
    X, 
    y, 
    cv=folds, # `folds` é a lista de GroupKFold que criamos anteriormente
    n_jobs=-1,
    method='predict_proba'
)

# as predicoes para a classe 1 (alto risco) estao na segunda coluna
y_pred_probas = y_pred_cv[:, 1]
# convertemos as probabilidades para classes (0 ou 1) usando um limiar de 0.5
y_pred_class = (y_pred_probas >= 0.5).astype(int)

# agora calculamos as metricas com base nas nossas predições controladas
accuracy = accuracy_score(y, y_pred_class)
report = classification_report(y, y_pred_class, target_names=["Classe 0", "Classe 1"])

print(f"Acurácia Média (cross_val_predict): {accuracy:.2%}")
print("\nRelatório de Classificação (cross_val_predict):")
print(report)
print("="*50)

print("\nAnálise SHAP e de Performance concluída com sucesso!")
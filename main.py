# main.py
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def feature_engineering_data(df_input):
    """Applies basic feature engineering to the dataframe."""
    df_processed = df_input.copy()
    df_processed['Data do pedido'] = pd.to_datetime(df_processed['Data do pedido'], errors='coerce')
    df_processed['Ano'] = df_processed['Data do pedido'].dt.year
    df_processed['Mes'] = df_processed['Data do pedido'].dt.month
    return df_processed

# 1. Carregar os dados
try:
    df_raw = pd.read_csv("database/dataset_para_modelo.csv")
except FileNotFoundError:
    print("Erro: O arquivo 'dataset_para_modelo.csv' não foi encontrado. Verifique o caminho.")
    exit()

# 2. Feature Engineering
# df_engineered = feature_engineering_data(df_raw)

# 3. Remover colunas desnecessárias e preparar X, y
# df_model_input = df_engineered.drop(columns=['ID do pedido', 'Data do pedido', 'País'])

df_model_input = df_raw.copy()

X = df_model_input.drop("valor_venda_total", axis=1)
y = df_model_input["valor_venda_total"]

# 4. Identificar colunas numéricas e categóricas
numeric_features = [
    'price', 'freight_value', 'product_name_lenght', 'product_description_lenght',
    'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'review_score', 'payment_sequential', 'payment_installments',
    'tempo_entrega_dias', 'tempo_estimado_dias', 'atraso_na_entrega_dias',
    'compra_dia_da_semana', 'compra_mes', 'percentual_frete'
]
categorical_features = [
    # 'seller_city',
    'seller_state',
    # 'customer_city',
    'customer_state',
    'payment_type',
    'product_category_name_english'
]

# 5. Criar o pré-processador com ColumnTransformer
# StandardScaler para numéricas, OneHotEncoder para categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
    ],
    remainder='drop' #descarta as colunas desnecessárias
)

# 6. Criar o pipeline com pré-processador e modelo
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# 7. Embrulhar o pipeline com TransformedTargetRegressor
regr_trans = TransformedTargetRegressor(
    regressor=pipeline_rf,
    func=np.log1p,
    inverse_func=np.expm1
)

# 8. Definir o grid de parâmetros para GridSearchCV
# Parâmetros do RandomForestRegressor são prefixos com 'model__'
param_grid = {
    'regressor__model__n_estimators': [100, 200],
    'regressor__model__max_depth': [10, 20, 30],
    'regressor__model__min_samples_leaf': [1, 2, 4],
    'regressor__model__min_samples_split': [2, 5, 10],
    'regressor__model__max_features': ['sqrt', 'log2']
}

# 9. Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Otimização com GridSearchCV
grid_search = GridSearchCV(
    estimator=regr_trans,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

try:
    grid_search.fit(X_train, y_train)
except Exception as e:
    print(f"Erro durante o GridSearchCV: {e}")
    exit()

best_pipeline = grid_search.best_estimator_

# 11. Avaliação do modelo otimizado no conjunto de teste
y_pred_test = best_pipeline.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
best_params = grid_search.best_params_

print(f"\n--- Resultados da Avaliação no Conjunto de Teste ---")
print(f"MSE (Teste): {mse_test:.2f}")
print(f"R² (Teste): {r2_test:.2f}")
print("Melhores parâmetros encontrados:", best_params)

# 12. Salvar métricas e melhores parâmetros
metrics_output = {
    "mse_teste": mse_test,
    "r2_teste": r2_test,
    "best_params": best_params
}
try:
    with open("data/model_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=4)
    print("\nMétricas do modelo salvas em model_metrics.json")
except IOError:
    print("Erro ao salvar as métricas do modelo.")

# 13. Exportar o pipeline completo
try:
    joblib.dump(best_pipeline, "data/modelo_vendas.pkl")
    print("Pipeline completo salvo como modelo_vendas.pkl")
except Exception as e:
    print(f"Erro ao salvar o pipeline: {e}")

# 14. (Opcional) Salvar nomes das features após o pré-processamento para referência
# Isso serve para interpretar as feature_importances no dashboard
try:
    # Acessamos o pipeline DENTRO do TransformedTargetRegressor através do atributo .regressor_
    preprocessor_step = best_pipeline.regressor_.named_steps['preprocessor']
    feature_names_out = preprocessor_step.get_feature_names_out()

    joblib.dump(list(feature_names_out), "data/encoders.pkl")  # Salva como lista
    print("Nomes das features processadas salvos em encoders.pkl")
except Exception as err:
    print(f"Erro ao salvar nomes das features processadas: {err}")

# Adicione no final de main.py, antes do "Script concluído."
print("\nGerando gráfico de diagnóstico...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue', label='Previsões vs. Real')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2, label='Linha Perfeita (y=x)')
plt.xlabel("Valor Real da Venda")
plt.ylabel("Valor Previsto da Venda")
plt.title("Valor Real vs. Valor Previsto no Conjunto de Teste")
plt.legend()
plt.grid(True)
plt.savefig("img/diagnostico_previsoes.png") # Salva a imagem no disco
print("Gráfico 'diagnostico_previsoes.png' salvo na pasta img.")

print("\nScript main.py concluído.")
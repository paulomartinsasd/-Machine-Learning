import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configurações de Estilo para os Gráficos ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

# --- Carregamento dos Dados ---
PROCESSED_DATA_PATH = os.path.join('data_processed', 'olist_dataset_completo.csv')

print(f"Carregando dataset de '{PROCESSED_DATA_PATH}'...")
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    # Converter colunas de data carregadas como texto
    for col in ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    print("Dataset carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado. Execute 'preparar_dados.py' primeiro.")
    exit()

# --- 1. Análise da Variável Alvo (payment_value) ---
print("\nGerando análise da variável alvo...")
plt.figure(figsize=(14, 6))
plt.suptitle("Análise da Variável Alvo (Valor do Pagamento)", fontsize=16)

plt.subplot(1, 2, 1)
sns.histplot(df['payment_value'], kde=True, bins=50)
plt.title("Distribuição do Valor do Pagamento")
plt.xlabel("Valor do Pagamento (R$)")
plt.ylabel("Frequência")

# Para melhor visualização, plotamos o log do valor
plt.subplot(1, 2, 2)
sns.histplot(df['payment_value'].apply(lambda x: np.log1p(x)), kde=True, bins=50, color='green')
plt.title("Distribuição do Log do Valor do Pagamento")
plt.xlabel("Log(1 + Valor do Pagamento)")
plt.ylabel("Frequência")

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()


# --- 2. Análise de Features Categóricas ---
print("\nGerando análise das features categóricas...")

# Top 15 Categorias de Produtos Mais Vendidas
plt.figure(figsize=(12, 8))
sns.countplot(y='product_category_name_english', data=df, order=df['product_category_name_english'].value_counts().iloc[:15].index)
plt.title('Top 15 Categorias de Produtos Mais Vendidas')
plt.xlabel('Contagem')
plt.ylabel('Categoria do Produto')
plt.tight_layout()
plt.show()

# Distribuição do Valor do Pagamento por Tipo de Pagamento
plt.figure(figsize=(12, 7))
sns.boxplot(x='payment_type', y='payment_value', data=df)
plt.title('Valor do Pagamento por Tipo de Pagamento')
plt.xlabel('Tipo de Pagamento')
plt.ylabel('Valor do Pagamento (R$)')
# Limitando o eixo y para melhor visualização dos boxplots (removendo outliers extremos do gráfico)
plt.ylim(0, 1000)
plt.show()


# --- 3. Análise de Features Numéricas ---
print("\nGerando análise das features numéricas...")

# Selecionar apenas algumas colunas numéricas importantes para a matriz de correlação
numeric_cols_corr = [
    'payment_value', 'price', 'freight_value', 'review_score',
    'payment_installments', 'product_weight_g', 'product_photos_qty'
]
correlation_matrix = df[numeric_cols_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação entre Features Numéricas')
plt.show()


# --- 4. Análise Temporal ---
print("\nGerando análise temporal...")

# Agrupar vendas por mês
df_temporal = df.set_index('order_purchase_timestamp')
vendas_por_mes = df_temporal.resample('ME').agg({'payment_value': 'sum'})

plt.figure(figsize=(14, 7))
vendas_por_mes['payment_value'].plot(kind='line', marker='o')
plt.title('Evolução do Valor Total de Vendas por Mês')
plt.xlabel('Mês da Compra')
plt.ylabel('Valor Total Vendido (R$)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\nAnálise concluída!")
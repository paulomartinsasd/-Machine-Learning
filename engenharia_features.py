import pandas as pd
import os

# Carregar o dataset completo que criamos
try:
    df = pd.read_csv('data_processed/olist_dataset_completo.csv')
    print("Dataset completo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'olist_dataset_completo.csv' não encontrado. Execute o script 'preparar_dados.py' primeiro.")
    exit()

# 1.1 Informações Gerais
print("\n--- Informações Gerais do Dataset ---")
df.info()

# 1.2 Verificar Dados Faltantes (Missing Values)
print("\n--- Contagem de Dados Faltantes por Coluna ---")
print(df.isnull().sum())

# 1.3 Corrigir Tipos de Dados (Especialmente Datas)
# As colunas de data são carregadas como texto, precisamos convertê-las.
cols_de_data = [
    'shipping_limit_date', 'review_creation_date', 'review_answer_timestamp',
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in cols_de_data:
    df[col] = pd.to_datetime(df[col], errors='coerce') # 'coerce' transforma erros em NaT (Not a Time)

print("\nTipos de dados corrigidos.")
df.info() # Verifique que as colunas de data agora são datetime64[ns]

print("\n--- Iniciando Engenharia de Features ---")

# 2.1 Definindo a Variável Alvo (Target)
# 'payment_value' é um ótimo candidato para o que queremos prever.
df.rename(columns={'payment_value': 'valor_venda_total'}, inplace=True)

# 2.2 Features a partir de Datas
# Calcular a diferença de tempo (em dias)
df['tempo_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['tempo_estimado_dias'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
df['atraso_na_entrega_dias'] = df['tempo_entrega_dias'] - df['tempo_estimado_dias']
df['atraso_na_entrega_dias'] = df['atraso_na_entrega_dias'].apply(lambda x: max(0, x)) # Se não houve atraso, o valor é 0

# Extrair componentes da data da compra
df['compra_dia_da_semana'] = df['order_purchase_timestamp'].dt.dayofweek # Segunda=0, Domingo=6
df['compra_mes'] = df['order_purchase_timestamp'].dt.month

# 2.3 Features a partir de Informações do Produto
# O dataset já tem peso, volume, etc. Vamos garantir que não haja nulos.
df['product_weight_g'].fillna(df['product_weight_g'].median(), inplace=True)
# Faça o mesmo para product_length_cm, etc.

# 2.4 Features a partir de Informações de Frete e Vendedor
df['percentual_frete'] = df['freight_value'] / df['valor_venda_total']

# 2.5 (Avançado) Calcular distância entre Cliente e Vendedor
# Esta é uma feature poderosa, mas mais complexa.
# Requer o arquivo de geolocalização e um pouco mais de processamento.
# (Deixaremos como um próximo passo avançado para não complicar agora)

print("Engenharia de features concluída!")

# 3.1 Definir a variável alvo (y) e remover linhas onde ela é nula
df.dropna(subset=['valor_venda_total'], inplace=True)
y = df['valor_venda_total']

# 3.2 Selecionar as colunas que serão as features (X)
# Removemos IDs, datas originais, e outras colunas que não devem entrar no modelo
colunas_para_remover = [
    'order_id', 'customer_id', 'order_item_id', 'product_id', 'seller_id',
    'customer_unique_id', 'order_status', 'shipping_limit_date',
    'review_id', 'review_comment_title', 'review_comment_message', 'review_creation_date',
    'review_answer_timestamp', 'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date', 'product_category_name', # Usaremos a versão em inglês
    'customer_zip_code_prefix', 'seller_zip_code_prefix' # CEPs têm muitas categorias, melhor usar cidade/estado
]

df_modelo = df.drop(columns=colunas_para_remover)

# Remover a variável alvo do conjunto de features
df_modelo = df_modelo.drop(columns=['valor_venda_total'])

# Preencher quaisquer outros valores nulos restantes com uma estratégia simples
# Para numéricos, usar a mediana. Para categóricos, usar a moda (valor mais comum).
for col in df_modelo.select_dtypes(include='number').columns:
    df_modelo[col] = df_modelo[col].fillna(df_modelo[col].median())
for col in df_modelo.select_dtypes(include='object').columns:
    df_modelo[col] = df_modelo[col].fillna(df_modelo[col].mode()[0])

# 3.3 Salvar o dataset final pronto para o modelo
df_modelo['valor_venda_total'] = y # Adicionar a variável alvo de volta para referência
output_path = 'database'
final_para_modelo_path = os.path.join(output_path, 'dataset_para_modelo.csv')
df_modelo.to_csv(final_para_modelo_path, index=False)

print("\n--- Processamento Finalizado ---")
print(f"Dataset final pronto para modelagem salvo em: {final_para_modelo_path}")
print(f"O dataset final tem {df_modelo.shape[1]} colunas (incluindo o alvo).")
print("Colunas finais:", df_modelo.columns.tolist())
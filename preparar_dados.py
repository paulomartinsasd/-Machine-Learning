import pandas as pd
import os

# --- Passo 0: Definir o caminho e carregar todos os arquivos ---

# Este script deve estar na mesma pasta principal que a pasta 'database'.
data_path = 'database'

print("Iniciando o carregamento dos arquivos...")

try:
    customers = pd.read_csv(os.path.join(data_path, 'olist_customers_dataset.csv'))
    geolocation = pd.read_csv(os.path.join(data_path, 'olist_geolocation_dataset.csv'))
    order_items = pd.read_csv(os.path.join(data_path, 'olist_order_items_dataset.csv'))
    payments = pd.read_csv(os.path.join(data_path, 'olist_order_payments_dataset.csv'))
    reviews = pd.read_csv(os.path.join(data_path, 'olist_order_reviews_dataset.csv'))
    orders = pd.read_csv(os.path.join(data_path, 'olist_orders_dataset.csv'))
    products = pd.read_csv(os.path.join(data_path, 'olist_products_dataset.csv'))
    sellers = pd.read_csv(os.path.join(data_path, 'olist_sellers_dataset.csv'))
    translation = pd.read_csv(os.path.join(data_path, 'product_category_name_translation.csv'))
    print("Todos os arquivos foram carregados com sucesso!")
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique se a pasta '{data_path}' existe e contém todos os CSVs. Detalhes: {e}")
    exit()


# --- Passo 1: A Cadeia Principal de Merges ---

# Começamos com a tabela 'order_items', que contém os itens de cada pedido.
# Usaremos 'left' merge para garantir que manteremos todos os itens da tabela original.

print("\nIniciando a combinação das tabelas (merge)...")

# 1.1 Adicionar informações dos Pedidos (orders) aos Itens
# Chave: order_id
data = pd.merge(order_items, orders, on='order_id', how='left')

# 1.2 Adicionar informações dos Produtos (products)
# Chave: product_id
data = pd.merge(data, products, on='product_id', how='left')

# 1.3 Adicionar informações dos Vendedores (sellers)
# Chave: seller_id
data = pd.merge(data, sellers, on='seller_id', how='left')

# 1.4 Adicionar informações dos Clientes (customers)
# Chave: customer_id
data = pd.merge(data, customers, on='customer_id', how='left')

# 1.5 Adicionar informações das Avaliações (reviews)
# Um pedido pode ter múltiplas avaliações, então vamos pegar apenas a mais recente por pedido
reviews = reviews.sort_values('review_answer_timestamp').drop_duplicates('order_id', keep='last')
data = pd.merge(data, reviews, on='order_id', how='left')


# --- Passo 2: Lidando com Tabelas Especiais (Pagamentos e Tradução) ---

# 2.1 Processar e adicionar informações de Pagamentos (payments)
# Um pedido pode ter múltiplos pagamentos (ex: boleto + voucher).
# Vamos agregar os dados de pagamento por pedido antes de fazer o merge.
payments_agg = payments.groupby('order_id').agg({
    'payment_sequential': 'max',
    'payment_type': 'first', # Pega o primeiro tipo de pagamento
    'payment_installments': 'max',
    'payment_value': 'sum'
}).reset_index()

# Agora fazemos o merge com os dados de pagamento agregados
data = pd.merge(data, payments_agg, on='order_id', how='left')


# 2.2 Adicionar a Tradução das Categorias de Produtos
# Chave: product_category_name
data = pd.merge(data, translation, on='product_category_name', how='left')


# --- Passo 3: Inspeção e Salvamento do Dataset Final ---

print("\nMerge concluído!")
print(f"O DataFrame final tem {data.shape[0]} linhas e {data.shape[1]} colunas.")

print("\nExemplo de colunas no DataFrame final:")
print(data.columns.tolist())

print("\nAmostra dos dados finais:")
print(data.head())

# Salvar o DataFrame completo em um único arquivo CSV para uso futuro
output_path = 'data_processed'
if not os.path.exists(output_path):
    os.makedirs(output_path)

final_csv_path = os.path.join(output_path, 'olist_dataset_completo.csv')
data.to_csv(final_csv_path, index=False)

print(f"\nDataFrame completo salvo com sucesso em: {final_csv_path}")
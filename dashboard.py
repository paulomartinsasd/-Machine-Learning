import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from fpdf import FPDF

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard de Vendas", layout="wide")
st.title("📊 Dashboard de Análise e Previsão de Vendas")


# --- Funções de Carregamento ---

@st.cache_data
def load_processed_data(file_path):
    """Carrega o dataset final, já processado."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Erro: O dataset processado '{file_path}' não foi encontrado.")
        st.info("Por favor, execute o script 'engenharia_features.py' primeiro para gerar o arquivo.")
        return None


@st.cache_resource
def load_model_pipeline(file_path):
    """Carrega o pipeline do modelo treinado."""
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        st.error(f"Erro: O pipeline '{file_path}' não foi encontrado.")
        st.info("Por favor, execute o script 'main.py' para treinar o modelo e salvar o pipeline.")
        return None


@st.cache_data
def load_json_data(file_path):
    """Carrega arquivos JSON, como as métricas do modelo."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de métricas '{file_path}' não foi encontrado.")
        st.info("Por favor, execute o script 'main.py'.")
        return None


# --- Carregamento Principal dos Artefatos ---

# Definindo os caminhos corretos para os artefatos
PROCESSED_DATA_PATH = os.path.join('database', 'dataset_para_modelo.csv')
PIPELINE_PATH = os.path.join('data', 'modelo_vendas.pkl')
METRICS_PATH = os.path.join('data', 'model_metrics.json')
FEATURES_NAMES_PATH = os.path.join('data', 'encoders.pkl')

# Carregar tudo
df_processed = load_processed_data(PROCESSED_DATA_PATH)
pipeline_model = load_model_pipeline(PIPELINE_PATH)
model_metrics = load_json_data(METRICS_PATH)
try:
    feature_names = joblib.load(FEATURES_NAMES_PATH)
except FileNotFoundError:
    feature_names = None

# Parar a execução se os arquivos essenciais não forem carregados
if df_processed is None or pipeline_model is None or model_metrics is None:
    st.stop()

# --- Abas do Dashboard ---
aba1, aba2, aba3 = st.tabs(["🎯 Previsão e Análise do Modelo", "📄 Gerar Relatório PDF", "📄 Sobre o Projeto"])

# --- Aba 1: Previsão e Análise ---
with aba1:
    st.header("🎯 Análise e Simulação do Modelo")
    st.markdown("---")

    # Layout com duas colunas principais
    col_form, col_results = st.columns(2, gap="large")

    with col_form:
        st.subheader("🤖 Previsão de Venda Simplificada")
        st.write("Preencha apenas as informações mais importantes para prever o valor da venda.")

        # --- INÍCIO DA LÓGICA DO FORMULÁRIO SIMPLIFICADO ---

        # Usando um "Dictionary Literal" para criar o dicionário de inputs
        # Isso torna o código mais conciso e direto.
        input_data_usuario = {
            'price': st.number_input("Preço do Produto (R$)", min_value=0.0, value=120.0, step=10.0),

            'freight_value': st.number_input("Valor do Frete (R$)", min_value=0.0, value=20.0, step=5.0),

            'product_weight_g': st.number_input("Peso do Produto (em gramas)", min_value=0, value=1500, step=100),

            'product_category_name_english': st.selectbox(
                "Categoria do Produto",
                options=sorted(df_processed['product_category_name_english'].dropna().unique().tolist()),
                index=sorted(df_processed['product_category_name_english'].dropna().unique().tolist()).index(
                    'bed_bath_table')
            ),

            'customer_state': st.selectbox(
                "Estado do Cliente",
                options=sorted(df_processed['customer_state'].dropna().unique().tolist()),
                index=sorted(df_processed['customer_state'].dropna().unique().tolist()).index('SP')
            ),

            'payment_installments': st.slider("Número de Parcelas", min_value=1, max_value=24, value=3)
        }

        if st.button("Prever Valor da Venda", key="predict_button", type="primary"):
            # A lógica a partir daqui contínua a mesma, pois ela precisa
            # preencher as features 'ocultas' que o usuário não inseriu.

            features_do_modelo = [
                'price', 'freight_value', 'product_name_lenght', 'product_description_lenght',
                'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm',
                'product_width_cm', 'review_score', 'payment_sequential', 'payment_installments',
                'tempo_entrega_dias', 'tempo_estimado_dias', 'atraso_na_entrega_dias',
                'compra_dia_da_semana', 'compra_mes', 'percentual_frete',
                'seller_state', 'customer_state', 'payment_type', 'product_category_name_english'
            ]

            input_completo = {}
            for feature in features_do_modelo:
                if feature in input_data_usuario:
                    input_completo[feature] = input_data_usuario[feature]
                else:
                    if df_processed[feature].dtype == 'object':
                        input_completo[feature] = df_processed[feature].mode()[0]
                    else:
                        input_completo[feature] = df_processed[feature].median()

            if 'price' in input_completo and input_completo['price'] > 0:
                input_completo['percentual_frete'] = input_completo['freight_value'] / (
                            input_completo['price'] + input_completo['freight_value'])
            else:
                input_completo['percentual_frete'] = 0

            try:
                input_df = pd.DataFrame([input_completo])
                input_df = input_df[features_do_modelo]

                prediction = pipeline_model.predict(input_df)[0]

                st.success(f"**Valor Previsto da Venda: R$ {prediction:.2f}**")
                st.session_state.ultima_predicao = prediction
            except Exception as e:
                st.error(f"Erro ao realizar a previsão: {e}")
                st.session_state.ultima_predicao = 0.0

    with col_results:
        st.subheader("🎯 Desempenho do Modelo (em dados de teste)")

        r2_model = model_metrics.get("r2_teste", 0.0)
        mse_model = model_metrics.get("mse_teste", 0.0)
        rmse_model = np.sqrt(mse_model)

        st.metric("R² (R-squared)", f"{r2_model:.2%}")
        st.metric("RMSE (Erro Médio)", f"R$ {rmse_model:.2f}")
        st.info(
            "O R² indica a porcentagem da variação do valor da venda que o modelo consegue explicar. O RMSE indica o erro médio das previsões em Reais.")

        st.subheader("📊 Importância das Features")
        st.write("Quais informações o modelo considera mais importantes?")

        try:
            inner_pipeline = pipeline_model.regressor_
            actual_model = inner_pipeline.named_steps['model']

            if feature_names and hasattr(actual_model, 'feature_importances_'):
                importances_series = pd.Series(actual_model.feature_importances_, index=feature_names).sort_values(
                    ascending=False)
                # Salvar as top features para usar no relatório
                st.session_state.top_features = importances_series.head(10)
                st.bar_chart(st.session_state.top_features)
            else:
                st.warning("Não foi possível obter a importância das features.")
        except Exception as e:
            st.error(f"Erro ao gerar o gráfico de importância: {e}")

# --- Aba 2: Gerar Relatório PDF ---
with aba2:
    st.header("📄 Gerar Relatório Técnico em PDF")


    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Relatório Técnico - Projeto de Machine Learning Aplicado ao Varejo", 0, 1, "C")
            self.ln(5)

        def chapter_title(self, title):
            self.set_font("Arial", "B", 11)
            self.cell(0, 8, title, 0, 1, "L")
            self.ln(2)

        def chapter_body(self, body):
            self.set_font("Arial", "", 10)
            # Tratar encoding para FPDF que espera latin-1 por padrão
            try:
                body_encoded = body.encode('latin-1', 'replace').decode('latin-1')
            except (ValueError, KeyError):
                body_encoded = "Erro ao codificar texto para PDF."
            self.multi_cell(0, 7, body_encoded)
            self.ln()


    def gerar_pdf():
        pdf = PDF()
        pdf.add_page()

        pdf.chapter_title("1. Introdução")
        pdf.chapter_body(
            "Este relatório resume os resultados do projeto de Machine Learning aplicado ao varejo, com foco na previsão do valor de venda de produtos. Foi utilizado um pipeline de dados com o algoritmo Random Forest."
        )

        pdf.chapter_title("2. Avaliação do Modelo (em Dados de Teste)")
        r2_pdf = model_metrics.get("r2_teste", 0.0)
        mse_pdf = model_metrics.get("mse_teste", 0.0)
        rmse_pdf = np.sqrt(mse_pdf)
        pdf.chapter_body(f"- R-squared (R²): {r2_pdf:.2%}\n- Root Mean Squared Error (RMSE): R$ {rmse_pdf:.2f}")

        pdf.chapter_title("3. Principais Variáveis Relevantes")
        top_features = st.session_state.get('top_features', pd.Series(dtype='float64'))
        if not top_features.empty:
            body_text = ""
            for feat, val in top_features.items():
                body_text += f"- {feat}: {val:.3f}\n"
            pdf.chapter_body(body_text.strip())
        else:
            pdf.chapter_body(
                "A importância das variáveis ainda não foi calculada. Por favor, volte para a aba de análise.")

        pdf.chapter_title("4. Exemplo de Previsão Recente")
        ultima_predicao = st.session_state.get('ultima_predicao', 0.0)
        pdf.chapter_body(
            f"O valor de venda previsto com base nos últimos dados informados na aba de simulação foi de: R$ {ultima_predicao:.2f}"
        )

        pdf.chapter_title("5. Conclusão e Próximos Passos")
        pdf.chapter_body(
            "O modelo de Random Forest apresentou um desempenho robusto, explicando uma parcela significativa da variabilidade dos dados. Melhorias futuras podem incluir a experimentação com outros algoritmos (ex: Gradient Boosting) e a criação de mais features de interação entre variáveis."
        )

        return pdf.output(dest="S").encode("latin-1")


    st.write("Clique no botão abaixo para gerar um resumo do projeto e dos resultados em um arquivo PDF.")

    if st.button("Gerar e Baixar Relatório", key="pdf_button"):
        try:
            # Cria a pasta 'outros' se ela não existir
            report_dir = "outros"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            pdf_bytes = gerar_pdf()
            st.download_button(
                label="📥 Clique aqui para baixar o PDF",
                data=pdf_bytes,
                file_name="outros/relatorio_tecnico_dinamico.pdf",
                mime="application/pdf"
            )
            st.success("Seu relatório está pronto para ser baixado!")
        except Exception as e:
            st.error(f"Erro ao gerar o PDF: {e}")


# --- Aba 3: Sobre o Projeto ---
with aba3:
    st.header("📄 Sobre Este Projeto")
    st.markdown("""
    Este dashboard é a interface de um projeto completo de Machine Learning para prever o valor de vendas no e-commerce.

    **O fluxo do projeto incluiu:**
    1.  **Coleta de Dados:** Utilização de 9 arquivos do dataset público da Olist.
    2.  **Preparação dos Dados:** Combinação (merge) das tabelas, limpeza de dados e criação de um dataset unificado.
    3.  **Engenharia de Features:** Criação de novas variáveis preditivas, como tempo de entrega, dia da semana da compra, etc.
    4.  **Treinamento do Modelo:**
        * Uso de um **Pipeline** para pré-processar os dados de forma robusta (padronizando números e codificando categorias).
        * Aplicação de **`TransformedTargetRegressor`** para normalizar a variável alvo, melhorando a estabilidade do modelo.
        * Treinamento de um modelo **`RandomForestRegressor`**.
        * Otimização de hiperparâmetros com **`GridSearchCV`**.
    5.  **Dashboard Interativo:** Esta interface, criada com Streamlit, permite que os usuários interajam com o modelo final, façam previsões e visualizem seus resultados.
    """)


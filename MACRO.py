import streamlit as st
import numpy as np
import pandas as pd
import openpyxl
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

file_path = "C:\\Users\\sam_r\\Desktop\\CODE\\PYTHON\\STREAMLIT\\MACRO3\\DADOS.xlsx"

df_country_and_indicators = pd.read_excel(file_path, sheet_name="Planilha1")
indicadores = df_country_and_indicators["INDICATOR"]
paises = df_country_and_indicators["Country Name"]

df_metadata = pd.read_excel(file_path, sheet_name="METADATA")

# OTIMIZAÇÃO PARA LEITURA DO ARQUIVO
def ler_arquivo_excel(file_path):
    df = pd.read_excel(file_path, sheet_name="DADOS_FILTRADOS")
    df['YEAR'] = pd.to_datetime(df['YEAR'], format='%Y')
    df['YEAR'] += pd.offsets.DateOffset(month=1, day=1)
    return df

st.title('ATIVIDADE MACRO III')
st.header("ANÁLISE DE VARIÁVEIS MACROECONÔMICAS")
st.markdown("Realizado por KAMILLA e SAMUEL")
st.write("")
st.write("")

with st.sidebar:
   
    paises_selecionados = st.multiselect(
        "SELECIONE OS PAÍSES QUE DESEJA:", df_country_and_indicators["Country Name"].unique()
        )
    coluna_selecionada_1 = st.selectbox("Selecione a variável para Y", indicadores)
    coluna_selecionada_2 = st.selectbox("Selecione a variável para X", indicadores)


df = ler_arquivo_excel(file_path)

# 
anos_unicos = df['YEAR'].dt.year.unique()
if len(anos_unicos) > 1:
    min_ano = min(anos_unicos)
    max_ano = max(anos_unicos)
else:
    min_ano = max_ano = anos_unicos[0]

# CÁLCULO DE PERCENTIL PARA EIXO DE X e Y
percentile_low = 5
percentile_high = 95

x_percentile_low = np.percentile(df[coluna_selecionada_2], percentile_low)
x_percentile_high = np.percentile(df[coluna_selecionada_2], percentile_high)
y_percentile_low = np.percentile(df[coluna_selecionada_1], percentile_low)
y_percentile_high = np.percentile(df[coluna_selecionada_1], percentile_high)

x_quartiles = np.nanpercentile(df[coluna_selecionada_2], [25, 50, 75])
y_quartiles = np.nanpercentile(df[coluna_selecionada_1], [25, 50, 75])

multiplicador = 6.0

indicadores_selecionados = [coluna_selecionada_1, coluna_selecionada_2]


x_range = [x_quartiles[0] - multiplicador * (x_quartiles[2] - x_quartiles[0]), x_quartiles[2] + multiplicador * (x_quartiles[2] - x_quartiles[0])]
y_range = [y_quartiles[0] - multiplicador * (y_quartiles[2] - y_quartiles[0]), y_quartiles[2] + multiplicador * (y_quartiles[2] - y_quartiles[0])]


st.write("")

st.subheader("GRÁFICO ANIMADO")



df_filtrado = df[df["Country Name"].isin(paises_selecionados)]

if coluna_selecionada_1 != coluna_selecionada_2:
    try:
        tempo = st.checkbox(f'Analisar {coluna_selecionada_1} em relação ao ano')
        tempo2 = st.checkbox(f'Analisar {coluna_selecionada_2} em relação ao ano')
    except:
        st.write("Por favor, selecione os anos antes!")
    st.write("")
    log_x_all = st.checkbox("Deseja utilizar log de X?")
    log_y_all = st.checkbox("Deseja utilizar log de Y?")
    
    
    if tempo is True:
        fig2 = px.scatter(df_filtrado, x="YEAR", y=coluna_selecionada_1, title=f'Gráfico de Dispersão ({coluna_selecionada_1} vs ANO)', animation_frame="YEAR", text="Country Code", color='Country Code', symbol='Country Code',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram', 'hexagon'])
    elif tempo2 is True:
        fig2 = px.scatter(df_filtrado, x="YEAR", y=coluna_selecionada_2, title=f'Gráfico de Dispersão ({coluna_selecionada_2} vs ANO)', animation_frame="YEAR", text="Country Code", color='Country Code', symbol='Country Code',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram', 'hexagon'])
    elif tempo is False and tempo2 is False:
        fig2 = px.scatter(df_filtrado, x=coluna_selecionada_2, y=coluna_selecionada_1, title=f'Gráfico de Dispersão ({coluna_selecionada_2} vs {coluna_selecionada_1})', animation_frame="YEAR",text="Country Code", color='Country Code', symbol='Country Code',
                        color_discrete_sequence=px.colors.qualitative.Set1, range_x=x_range, range_y=y_range, log_x=log_x_all, log_y=log_y_all,
                        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram', 'hexagon'])

    else:
        st.write("Não é possível selecionar os dois ao mesmo tempo! \n Por favor, selecione apenas uma ou nenhuma das opções")
    
    fig2.update_traces(textposition='top center')
    fig2.update_layout(width=800, height=600)

    st.plotly_chart(fig2)

st.subheader("Análise de Regressão para um período específico")


try:
    anos_unicos = df_filtrado['YEAR'].dt.year.unique()

    ano_inicial = st.selectbox("Selecione o ano INICIAL", anos_unicos)
    ano_inicial_datetime = pd.to_datetime(str(ano_inicial), format='%Y')

    anos_acima = [ano for ano in anos_unicos if ano >= ano_inicial_datetime.year]

    ano_final = st.selectbox("Selecione o ano FINAL", anos_acima)

    st.write("")
    media_periodo = st.checkbox("Deseja utilizar a média do período?")
    st.write("")

except:
    st.write("Por favor! selecione os PAÍSES e INDICADORES à esquerda primeiro.")





if coluna_selecionada_1 != coluna_selecionada_2:
    df_entre_anos = df_filtrado[(df_filtrado['YEAR'] >= pd.to_datetime(str(ano_inicial), format="%Y")) & (df_filtrado['YEAR'] <= pd.to_datetime(str(ano_final), format="%Y"))]
    
    
    if media_periodo:
        colunas_selecionadas = ["Country Name", "Country Code", coluna_selecionada_1, coluna_selecionada_2]
        df_entre_anos_final = df_entre_anos[colunas_selecionadas]
        media_total = df_entre_anos_final.groupby(['Country Name', "Country Code"])[[coluna_selecionada_1, coluna_selecionada_2]].mean().reset_index()
        info_df = media_total.describe()
        st.write(info_df)

                # TRATAMENTO DE VALORES FALTANTES
        if media_total.isnull().values.any():
            info_df = media_total.describe()
            st.write("")
            st.write("Existem valores faltantes.")
            st.write("Tabela com Contagem de Valores Faltantes:")
            st.write(media_total[colunas_selecionadas].isnull().sum())

            tratamento = st.radio("Escolha como tratar os valores faltantes:", ["REMOÇÃO", "Preencher com a Média", "Preencher com a Mediana"])

            if tratamento == "REMOÇÃO":
                media_total = media_total.dropna()
                st.write("Linhas com valores faltantes foram removidas.")

            elif tratamento == "Preencher com a Média":
                imputer = SimpleImputer(strategy='mean')
                media_total[[coluna_selecionada_1, coluna_selecionada_2]] = imputer.fit_transform(media_total[[coluna_selecionada_1, coluna_selecionada_2]])
                st.write("Valores faltantes foram preenchidos com a média.")
            elif tratamento == "Preencher com a Mediana":
                imputer = SimpleImputer(strategy='median')
                media_total[[coluna_selecionada_1, coluna_selecionada_2]] = imputer.fit_transform(media_total[[coluna_selecionada_1, coluna_selecionada_2]])
                st.write("Valores faltantes foram preenchidos com a mediana.")
                    
        
        fig3 = px.scatter(media_total, x=coluna_selecionada_2, y=coluna_selecionada_1, title=f'Gráfico de Dispersão ({coluna_selecionada_1} vs {coluna_selecionada_2})', text="Country Code", color='Country Code', symbol='Country Code',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram', 'hexagon'])
        fig3.update_traces(textposition='top center')
   

        X = media_total[coluna_selecionada_2]
        Y = media_total[coluna_selecionada_1]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        
        equacao_regressao = f"Equação de Regressão: {model.params['const']:.2f} + {model.params[coluna_selecionada_2]:.2f} * {coluna_selecionada_2}"
        fig3.add_trace(go.Scatter(x=media_total[coluna_selecionada_2], y=model.predict(X), mode='lines', name='Linha de Regressão'))
        
        st.write("")
        st.plotly_chart(fig3)
        st.write(equacao_regressao)
        st.write("")
        st.text(model.summary())

    else:
        colunas_selecionadas = ["Country Name", "Country Code", coluna_selecionada_1, coluna_selecionada_2]  
        df_entre_anos_final = df_entre_anos[colunas_selecionadas]
        # TRATAMENTO DE VALORES FALTANTES
        if df.isnull().values.any():
            info_df = df_entre_anos_final.describe()
            st.write(info_df)
            st.write("")
            st.write("Existem valores faltantes.")
            st.write("Tabela com Contagem de Valores Faltantes:")
            st.write(df_entre_anos_final.isnull().sum())

            tratamento = st.radio("Escolha como tratar os valores faltantes:", ["REMOÇÃO", "Preencher com a Média", "Preencher com a Mediana"])

            if tratamento == "REMOÇÃO":
                df_entre_anos_final = df_entre_anos_final.dropna()
                st.write("Linhas com valores faltantes foram removidas.")

            elif tratamento == "Preencher com a Média":
                imputer = SimpleImputer(strategy='mean')
                df_entre_anos_final[[coluna_selecionada_1, coluna_selecionada_2]] = imputer.fit_transform(df_entre_anos_final[[coluna_selecionada_1, coluna_selecionada_2]])
                st.write("Valores faltantes foram preenchidos com a média.")
            elif tratamento == "Preencher com a Mediana":
                imputer = SimpleImputer(strategy='median')
                df_entre_anos_final[[coluna_selecionada_1, coluna_selecionada_2]] = imputer.fit_transform(df_entre_anos_final[[coluna_selecionada_1, coluna_selecionada_2]])
                st.write("Valores faltantes foram preenchidos com a mediana.")

        fig4 = px.scatter(df_entre_anos_final, x=coluna_selecionada_2, y=coluna_selecionada_1, title=f'Gráfico de Dispersão ({coluna_selecionada_1} vs {coluna_selecionada_2})', text="Country Code", color='Country Code', symbol='Country Code',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'hexagram', 'hexagon'])
        fig4.update_traces(textposition='top center')

        X = df_entre_anos_final[coluna_selecionada_2]
        Y = df_entre_anos_final[coluna_selecionada_1]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()

        equacao_regressao = f"Equação de Regressão: {model.params['const']:.2f} + {model.params[coluna_selecionada_2]:.2f} * {coluna_selecionada_2}"
        fig4.add_trace(go.Scatter(x=df_entre_anos_final[coluna_selecionada_2], y=model.predict(X), mode='lines', name='Linha de Regressão'))

        st.write("")
        st.plotly_chart(fig4)
        st.write(equacao_regressao)
        st.write("")
        st.text(model.summary())
else: 
    st.write("Selecione colunas diferentes para os eixos x e y.")


try:
    st.title("METADATA")
    df_metadata = df_metadata[(df_metadata["Indicator Name"] == coluna_selecionada_1) | (df_metadata["Indicator Name"] == coluna_selecionada_2)]
    st.write(df_metadata)
except:
    pass

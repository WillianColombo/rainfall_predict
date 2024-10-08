import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import joblib

# Carregar o modelo treinado
model = joblib.load('/workspaces/rainfall_predict/modelo_treinado.pkl')  # Ajuste o caminho para o seu modelo
scaler = joblib.load('/workspaces/rainfall_predict/scaler_treinado.pkl')  # Ajuste o caminho para o seu scaler

st.title("Previsão de Precipitação")

# Upload do CSV
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type='csv')

if uploaded_file is not None:
    # Ler o arquivo CSV
    df_new = pd.read_csv(uploaded_file, sep=';')
    
    # Limpeza e formatação dos novos dados
    df_new['PRECIPITAO TOTAL, HORRIO (mm)'] = df_new['PRECIPITAO TOTAL, HORRIO (mm)'].str.replace(',', '.').astype(float)
    df_new['PRECIPITAO TOTAL, HORRIO (mm)'] = (df_new['PRECIPITAO TOTAL, HORRIO (mm)'] > 0).astype(int)
    
    for column in df_new.select_dtypes(include='object').columns:
        df_new[column] = df_new[column].str.replace(',', '.').astype(float, errors='ignore')

    # Remover colunas irrelevantes
    df_new = df_new.drop(columns=[
        'RADIACAO GLOBAL (Kj/m)', 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)', 'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (C)',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (C)', 'VENTO, RAJADA MAXIMA (m/s)',
        'Unnamed: 19', 'Data', 'Hora UTC', 'TEMPERATURA MNIMA NA HORA ANT. (AUT) (C)',
        'TEMPERATURA MXIMA NA HORA ANT. (AUT) (C)'
    ])
    df_new.dropna(inplace=True)

    # Separar features
    features_new = df_new.drop(columns=['PRECIPITAO TOTAL, HORRIO (mm)'])
    
    # Normalizar os novos dados
    X_new = scaler.transform(features_new)
    
    # Fazer predições
    predictions = model.predict(X_new)

    # Adicionar colunas de previsões e valor real ao dataframe
    df_new['Valor Real'] = df_new['PRECIPITAO TOTAL, HORRIO (mm)']  # Coluna com os valores reais
    df_new['Previsao Chuva'] = predictions  # Coluna com as previsões

    # Exibir o dataframe resultante
    st.write("Dados processados:")
    st.dataframe(df_new)

    # Mostrar o relatório de classificação
    report = classification_report(df_new['Valor Real'], predictions, output_dict=True)
    st.write("Relatório de Classificação:")
    st.text(classification_report(df_new['Valor Real'], predictions))

    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(df_new['Valor Real'], predictions, ax=ax)
    st.pyplot(fig)

    # Dashboard com gráficos
    st.subheader("Distribuição das Previsões")
    fig2, ax2 = plt.subplots()
    sb.countplot(data=df_new, x='Previsao Chuva', ax=ax2)
    st.pyplot(fig2)

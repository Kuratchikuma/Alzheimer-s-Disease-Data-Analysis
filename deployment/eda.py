import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def run():

    # Membuat Title
    st.title('Alzheimer\'s Disease Data Analysis')

    # Membuat subheader
    st.subheader('EDA untuk Analisa Penyakit Alzheimer')

    # Menampilkan pembuat page
    st.write('Made by Fahri')

    # Bold dan italic contoh teks
    st.write('**Analisa Data Alzheimer**')
    st.write('*Menggunakan dataset kesehatan Alzheimer*')

    # Membuat garis lurus
    st.markdown('---')

    # Load dataset
    df = pd.read_csv('alzheimers_disease_data.csv')
    st.dataframe(df)
    df = df.drop(columns=['DoctorInCharge'])

    # Visualisasi Distribusi Diagnosis Alzheimer
    st.write('#### Distribution of Alzheimer\'s Disease Diagnosis')
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x='Diagnosis', data=df)
    plt.title('Distribution of Alzheimer\'s Disease Diagnosis')
    st.pyplot(fig)

    # Heatmap Korelasi Antar Fitur
    st.write('#### Correlation Heatmap')
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    # Visualisasi Perbandingan Kasus Diabetes Berdasarkan Etnis
    st.write('#### Comparison of Diabetes Cases Across Ethnicities')
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x='Ethnicity', hue='Diabetes', data=df, palette='coolwarm')
    plt.title('Comparison of Diabetes Cases Across Ethnicities', fontsize=16)
    plt.xlabel('Ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Diabetes (0: No, 1: Yes)')
    st.pyplot(fig)

    # Rasio Penderita Diabetes per Etnis
    st.write('#### Diabetes Ratio per Ethnicity')
    ethnicity_diabetes_ratio = df.groupby('Ethnicity')['Diabetes'].mean()
    ethnicity_diabetes_ratio_df = pd.DataFrame({
        'Ethnicity': ethnicity_diabetes_ratio.index,
        'Diabetes Ratio': ethnicity_diabetes_ratio.values
    })
    fig = plt.figure(figsize=(8, 6))
    sns.barplot(x='Ethnicity', y='Diabetes Ratio', data=ethnicity_diabetes_ratio_df, palette='coolwarm')
    plt.title('Rasio Penderita Diabetes per Etnis', fontsize=16)
    plt.xlabel('Etnis (0: Caucasian, 1: African American, 2: Asian, 3: Lainnya)', fontsize=12)
    plt.ylabel('Rasio Diabetes', fontsize=12)
    plt.ylim(0, 1)
    st.pyplot(fig)

    # Distribusi Diagnosis berdasarkan Status Diabetes
    st.write('#### Distribution of Diagnosis by Diabetes Status')
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x='Diagnosis', hue='Diabetes', data=df)
    plt.title('Distribution of Diagnosis by Diabetes Status')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.legend(title='Diabetes', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Distribusi Diagnosis berdasarkan Riwayat Keluarga Alzheimer
    st.write('#### Distribution of Diagnosis by Family History of Alzheimer’s')
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x='Diagnosis', hue='FamilyHistoryAlzheimers', data=df)
    plt.title('Distribution of Diagnosis by Family History of Alzheimer’s')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.legend(title='Family History of Alzheimer’s', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    st.pyplot(fig)

if __name__ == '__main__':
    run()

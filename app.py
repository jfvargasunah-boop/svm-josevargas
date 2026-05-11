import os
import json
import joblib
import pandas as pd
import streamlit as st
from groq import Groq


st.set_page_config(page_title="Riesgo actuarial-Jose Francisco Vargas", layout="centered")
st.title("Predicción de riesgo actuarial-José Francisco Vargas Sierra")


@st.cache_resource
def cargar_modelo():
    # Metadata embebida en lugar de archivo JSON
    metadata = {
        "nombre_modelo": "K-means + SVM para riesgo actuarial",
        "tipo_modelo": "Clustering no supervisado + clasificación supervisada didáctica",
        "n_clusters": 3,
        "silhouette_score": 0.178,
        "variables_numericas": ["age", "bmi", "children", "charges"],
        "variables_categoricas": ["sex", "smoker", "region"],
        "mapa_riesgo": {"0": "Alto", "1": "Bajo", "2": "Medio"},
        "svm_kernels": ["linear", "poly", "rbf", "sigmoid"]
    }
    
    return None, metadata


@st.cache_data
def cargar_base():
    csv = "insurance_con_clusters.csv" if os.path.exists("insurance_con_clusters.csv") else "insurance.csv"
    return pd.read_csv(csv)


modelo, metadata = cargar_modelo()
df = cargar_base()
mapa = {int(k): v for k, v in metadata["mapa_riesgo"].items()}

st.caption(metadata["nombre_modelo"])

with st.form("datos"):
    col1, col2 = st.columns(2)

    age = col1.number_input("Edad", 18, 100, 35)
    sex = col2.selectbox("Sexo", sorted(df["sex"].unique()))
    bmi = col1.number_input("BMI", 10.0, 60.0, 28.0)
    children = col2.number_input("Hijos", 0, 10, 1)
    smoker = col1.selectbox("Fumador", sorted(df["smoker"].unique()))
    region = col2.selectbox("Región", sorted(df["region"].unique()))
    charges = st.number_input(
        "Cargos médicos estimados",
        0.0,
        100000.0,
        12000.0
    )

    enviar = st.form_submit_button("Evaluar")


if enviar:
    cliente = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges,
        }
    ])

    # Encontrar el cliente más similar en la base de datos para obtener su cluster
    from sklearn.preprocessing import LabelEncoder
    
    df_temp = df.copy() if "Cluster" not in df.columns else df.copy()
    
    # Si el dataframe tiene la columna Cluster, usarla. Si no, asignar por defecto
    if "Cluster" in df_temp.columns:
        # Codificar variables categóricas para comparación
        le_sex = LabelEncoder()
        le_smoker = LabelEncoder()
        le_region = LabelEncoder()
        
        df_temp["sex_encoded"] = le_sex.fit_transform(df_temp["sex"])
        df_temp["smoker_encoded"] = le_smoker.fit_transform(df_temp["smoker"])
        df_temp["region_encoded"] = le_region.fit_transform(df_temp["region"])
        
        cliente["sex_encoded"] = le_sex.transform([sex])[0]
        cliente["smoker_encoded"] = le_smoker.transform([smoker])[0]
        cliente["region_encoded"] = le_region.transform([region])[0]
        
        # Calcular distancia euclidiana
        features = ["age", "bmi", "children", "charges", "sex_encoded", "smoker_encoded", "region_encoded"]
        distances = ((df_temp[features] - cliente[features].values) ** 2).sum(axis=1) ** 0.5
        
        # Obtener el índice del cliente más similar
        idx_similar = distances.idxmin()
        cluster = int(df_temp.loc[idx_similar, "Cluster"])
    else:
        cluster = 1  # Valor por defecto
    
    riesgo = mapa.get(cluster, "No definido")

    st.subheader(f"Riesgo actuarial: {riesgo}")
    st.write(f"Cluster asignado: {cluster}")

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    if api_key:
        prompt = f"""
        Actúa como analista actuarial.

        Explica brevemente el resultado del modelo y brinda 3 recomendaciones prudentes,
        claras y profesionales para el usuario.

        Datos del cliente:
        - Edad: {age}
        - Sexo: {sex}
        - BMI: {bmi}
        - Hijos: {children}
        - Fumador: {smoker}
        - Región: {region}
        - Cargos médicos estimados: {charges}

        Resultado del modelo:
        - Cluster asignado: {cluster}
        - Nivel de riesgo actuarial: {riesgo}
        """

        try:
            client = Groq(api_key=api_key)

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista actuarial prudente, claro y profesional.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.4,
                max_tokens=500,
            )

            respuesta = completion.choices[0].message.content
            st.info(respuesta)

        except Exception as e:
            st.warning(f"No se pudo generar recomendación con Groq: {e}")

    else:
        st.warning("Agregue GROQ_API_KEY en los secretos de Streamlit.")


st.divider()
st.write("Vista rápida de la base principal")
st.dataframe(df.head(20), use_container_width=True)

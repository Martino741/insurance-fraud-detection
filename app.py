from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -------------------------------------------------
# 1. Chemins des fichiers
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "insurance_fraud_model.joblib"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed_claim_features.csv"


# -------------------------------------------------
# 2. Configuration de la page
# -------------------------------------------------
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Automobile Insurance Fraud Detection")
st.write(
    "Cette application identifie les déclarations d'assurance "
    "automobile qui nécessitent une vérification complémentaire."
)


# -------------------------------------------------
# 3. Charger le modèle sauvegardé
# -------------------------------------------------
@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Le modèle est introuvable. Exécute d'abord les cellules "
        "de sauvegarde du notebook."
    )
    st.stop()

artifact = load_model_artifact()

model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]
model_name = artifact["model_name"]

st.caption(f"Modèle utilisé : {model_name}")


# -------------------------------------------------
# 4. Charger les données
# -------------------------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


uploaded_file = st.file_uploader(
    "Importer un fichier CSV préparé",
    type=["csv"],
    help="Le fichier doit contenir les mêmes colonnes que processed_claim_features.csv."
)

if uploaded_file is not None:
    claims = pd.read_csv(uploaded_file)
else:
    if not DEFAULT_DATA_PATH.exists():
        st.error(
            "Le fichier processed_claim_features.csv est introuvable. "
            "Crée-le depuis le notebook."
        )
        st.stop()

    claims = load_default_data()


# -------------------------------------------------
# 5. Vérifier les colonnes du fichier
# -------------------------------------------------
missing_columns = [
    column for column in feature_names
    if column not in claims.columns
]

if missing_columns:
    st.error(
        "Le fichier importé ne possède pas toutes les colonnes attendues."
    )
    st.write("Colonnes manquantes :", missing_columns)
    st.stop()

# Conserver les colonnes dans le même ordre que pendant l'entraînement
claims = claims[feature_names]


# -------------------------------------------------
# 6. Choisir un dossier
# -------------------------------------------------
st.subheader("Analyser une déclaration")

claim_index = st.number_input(
    "Numéro de la déclaration",
    min_value=0,
    max_value=len(claims) - 1,
    value=0,
    step=1
)

selected_claim = claims.iloc[[claim_index]]

if st.button("Analyser la déclaration"):
    selected_claim_scaled = scaler.transform(selected_claim)

    prediction = model.predict(selected_claim_scaled)[0]

    if prediction == 1:
        st.error(
            "⚠️ Risque élevé : cette déclaration doit être examinée "
            "par un enquêteur."
        )
    else:
        st.success(
            "✅ Risque faible : aucune suspicion forte détectée "
            "par le modèle."
        )

    st.subheader("Informations du dossier analysé")

    display_columns = [
        "age",
        "months_as_customer",
        "policy_annual_premium",
        "total_claim_amount",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "number_of_vehicles_involved",
        "witnesses"
    ]

    available_display_columns = [
        column for column in display_columns
        if column in selected_claim.columns
    ]

    st.dataframe(
        selected_claim[available_display_columns],
        use_container_width=True
    )


# -------------------------------------------------
# 7. Analyser toutes les déclarations
# -------------------------------------------------
st.divider()
st.subheader("Analyse de toutes les déclarations")

if st.button("Analyser tous les dossiers"):
    claims_scaled = scaler.transform(claims)

    predictions = model.predict(claims_scaled)

    results = claims.copy()
    results["fraud_prediction"] = predictions
    results["risk_label"] = results["fraud_prediction"].map({
        0: "Risque faible",
        1: "À examiner"
    })

    st.write("Résultat global :")
    st.bar_chart(results["risk_label"].value_counts())

    st.dataframe(
        results[["fraud_prediction", "risk_label"]],
        use_container_width=True
    )

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Télécharger les prédictions",
        data=csv,
        file_name="insurance_fraud_predictions.csv",
        mime="text/csv"
    )


st.caption(
    "Le modèle est un outil de priorisation des dossiers. "
    "Une validation humaine reste nécessaire."
)
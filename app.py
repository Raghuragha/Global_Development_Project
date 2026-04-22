import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Load models ---
try:
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    kmeans_model = joblib.load("kmeans.joblib")
    columns = joblib.load("columns.joblib")  # correct training columns
except FileNotFoundError:
    st.error("Missing model files. Ensure all .joblib files are uploaded.")
    st.stop()

st.title("Global Development Clustering App")
st.write("Upload dataset to classify countries into clusters")

uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # --- Read file ---
    try:
        if uploaded_file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"File reading error: {e}")
        st.stop()

    st.write("### Original Data")
    st.dataframe(df_uploaded.head())

    # --- Save Country column ---
    country_names = df_uploaded['Country'].copy() if 'Country' in df_uploaded.columns else None

    # --- Drop non-feature column ---
    df_processed = df_uploaded.drop('Country', axis=1, errors='ignore').copy()

    # =========================================================
    # STEP 1: ALIGN COLUMNS FIRST (VERY IMPORTANT)
    # =========================================================
    original_feature_cols = columns

    # Add missing columns
    for col in original_feature_cols:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # Remove extra columns & reorder
    df_processed = df_processed[original_feature_cols]

    # =========================================================
    #  STEP 2: CLEAN DATA (AFTER ALIGNMENT)
    # =========================================================
    for col in df_processed.columns:
        temp = df_processed[col].astype(str)\
            .str.replace('$', '', regex=False)\
            .str.replace(',', '', regex=False)\
            .str.strip()

        if temp.str.contains('%').any():
            df_processed[col] = pd.to_numeric(
                temp.str.replace('%', '', regex=False),
                errors='coerce'
            ) / 100
        else:
            df_processed[col] = pd.to_numeric(temp, errors='coerce')

    # =========================================================
    # DEBUG (OPTIONAL - REMOVE LATER)
    # =========================================================
    st.write("Expected features:", len(original_feature_cols))
    st.write("Input shape:", df_processed.shape)

    # =========================================================
    # STEP 3: SCALING
    # =========================================================
    try:
        X_scaled = scaler.transform(df_processed)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # =========================================================
    # STEP 4: HANDLE MISSING VALUES (SAFE)
    # =========================================================
    # NOTE: Ideally load saved imputer
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)

    # =========================================================
    # STEP 5: PCA TRANSFORMATION
    # =========================================================
    try:
        X_pca = pca.transform(X_imputed)
    except Exception as e:
        st.error(f"PCA error: {e}")
        st.stop()

    # =========================================================
    # STEP 6: PREDICTION
    # =========================================================
    try:
        clusters = kmeans_model.predict(X_pca)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # --- Cluster mapping ---
    cluster_names = {
        0: "Developed",
        1: "Developing",
        2: "Underdeveloped"
    }

    df_uploaded['Cluster'] = clusters
    df_uploaded['Cluster_Name'] = df_uploaded['Cluster'].map(cluster_names)

    if country_names is not None:
        df_uploaded['Country'] = country_names

    # =========================================================
    # RESULTS
    # =========================================================
    st.write("### Cluster Results")

    if 'Country' in df_uploaded.columns:
        st.dataframe(df_uploaded[['Country', 'Cluster_Name']])
    else:
        st.dataframe(df_uploaded[['Cluster_Name']])

    # =========================================================
    # VISUALIZATION
    # =========================================================
    st.write("### Cluster Visualization")

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("K-Means Clusters")

    handles, _ = scatter.legend_elements()
    unique_clusters = sorted(np.unique(clusters))
    labels = [cluster_names.get(c, f"Cluster {c}") for c in unique_clusters]
    ax.legend(handles, labels, title="Clusters")

    st.pyplot(fig)
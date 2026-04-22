import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Global Dev Clustering", layout="wide")

# =========================
# CUSTOM DARK UI
# =========================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.block-container {
    padding: 2rem;
}
.card {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.metric {
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌍 Global Dev Clustering")
st.sidebar.write("Unsupervised ML Project")

uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    kmeans = joblib.load("kmeans.joblib")
    columns = joblib.load("columns.joblib")
    return scaler, pca, kmeans, columns

try:
    scaler, pca, model, columns = load_models()
except:
    st.error("❌ Missing model files (.joblib)")
    st.stop()

# =========================
# MAIN APP
# =========================
if uploaded_file:

    # =========================
    # READ FILE
    # =========================
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "Country" not in df.columns:
        st.error("Dataset must contain 'Country' column")
        st.stop()

    country_names = df["Country"]

    # =========================
    # CLEAN DATA (NUMERIC SAFE)
    # =========================
    df_clean = df.drop("Country", axis=1).copy()

    # Align columns
    for col in columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    df_clean = df_clean[columns]

    # Convert to numeric
    for col in df_clean.columns:
        temp = df_clean[col].astype(str)\
            .str.replace('$', '', regex=False)\
            .str.replace(',', '', regex=False)\
            .str.strip()

        if temp.str.contains('%').any():
            df_clean[col] = pd.to_numeric(
                temp.str.replace('%', '', regex=False),
                errors='coerce'
            ) / 100
        else:
            df_clean[col] = pd.to_numeric(temp, errors='coerce')

    # =========================
    # TRANSFORM PIPELINE
    # =========================
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")

    X_scaled = scaler.transform(df_clean)
    X_imputed = imputer.fit_transform(X_scaled)
    X_pca = pca.transform(X_imputed)

    clusters = model.predict(X_pca)

    # Add cluster to df (for display)
    df["Cluster"] = clusters

    # =========================
    # COUNTRY SELECTOR
    # =========================
    st.markdown("## 📊 Per-Country Detail Card")

    selected_country = st.selectbox(
        "Select country to inspect:",
        country_names
    )

    # Get row index safely
    row_index = df[df["Country"] == selected_country].index[0]

    # Clean numeric row
    row_clean = df_clean.iloc[row_index]

    # =========================
    # HEADER CARD
    # =========================
    st.markdown(f"""
    <div class="card">
        <h3>🌐 {selected_country}</h3>
        <p>Cluster assigned: <b>{int(df.loc[row_index, 'Cluster'])}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # METRICS GRID
    # =========================
    st.markdown("### Key Indicators")

    cols = st.columns(4)

    for i, col_name in enumerate(df_clean.columns[:8]):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card">
                <p>{col_name}</p>
                <div class="metric">{round(row_clean[col_name], 2)}</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CLUSTER MEAN COMPARISON (FIXED)
    # =========================
    st.markdown("## 📉 Country vs Cluster Mean")

    cluster_id = df.loc[row_index, "Cluster"]

    # Use CLEAN DATA (IMPORTANT FIX)
    cluster_data = df_clean.copy()
    cluster_data["Cluster"] = clusters

    cluster_mean = cluster_data[cluster_data["Cluster"] == cluster_id].mean()

    # =========================
    # PLOT
    # =========================
    features_to_plot = df_clean.columns[:5]

    fig, ax = plt.subplots()

    ax.bar(features_to_plot, row_clean[features_to_plot], label="Country")
    ax.bar(features_to_plot, cluster_mean[features_to_plot], alpha=0.5, label="Cluster Mean")

    plt.xticks(rotation=45)
    plt.legend()

    st.pyplot(fig)

else:
    st.info("⬅️ Upload dataset to begin")

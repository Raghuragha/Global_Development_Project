import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pycountry
from sklearn.impute import SimpleImputer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Global Dev Clustering", layout="wide")

# =========================
# DARK UI
# =========================
st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
.block-container {padding: 1.5rem;}
.card {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.metric {
    font-size: 20px;
    font-weight: bold;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FLAG FUNCTION
# =========================
def get_flag(country_name):
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        code = country.alpha_2
        return "".join(chr(127397 + ord(c)) for c in code)
    except:
        return ""

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌍 Global Dev Clustering")
st.sidebar.caption("Unsupervised ML Project")

uploaded_file = st.sidebar.file_uploader("📂 Upload dataset", type=["csv", "xlsx"])

menu = st.sidebar.radio("📊 Navigation", [
    "Overview & EDA",
    "Feature Analysis",
    "Clustering Models",
    "Model Comparison",
    "Country Explorer"
])

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

    # LOAD DATA
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "Country" not in df.columns:
        st.error("Dataset must contain 'Country'")
        st.stop()

    country_names = df["Country"]

    # =========================
    # COUNTRY FILTER
    # =========================
    st.sidebar.markdown("### 🌐 Select Country")

    country_list = sorted(country_names.unique())

    selected_country = st.sidebar.selectbox(
        "Choose Country",
        ["All Countries"] + country_list,
        format_func=lambda x: f"{get_flag(x)} {x}" if x != "All Countries" else x
    )

    # =========================
    # CLEAN DATA
    # =========================
    df_clean = df.drop("Country", axis=1).copy()

    for col in columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    df_clean = df_clean[columns]

    for col in df_clean.columns:
        temp = df_clean[col].astype(str)\
            .str.replace('$', '', regex=False)\
            .str.replace(',', '', regex=False)

        if temp.str.contains('%').any():
            df_clean[col] = pd.to_numeric(
                temp.str.replace('%', '', regex=False),
                errors='coerce'
            ) / 100
        else:
            df_clean[col] = pd.to_numeric(temp, errors='coerce')

    # =========================
    # IMPUTATION (NO NaN)
    # =========================
    imputer = SimpleImputer(strategy="mean")
    df_clean[:] = imputer.fit_transform(df_clean)

    # =========================
    # TRANSFORM
    # =========================
    X_scaled = scaler.transform(df_clean)
    X_pca = pca.transform(X_scaled)

    clusters = model.predict(X_pca)
    df["Cluster"] = clusters

    # =========================
    # 🔥 CLUSTER LABELS (FIXED)
    # =========================
    cluster_data = df_clean.copy()
    cluster_data["Cluster"] = clusters

    # Ensure GDP is numeric
    cluster_data["GDP"] = pd.to_numeric(cluster_data["GDP"], errors="coerce")

    cluster_means = cluster_data.groupby("Cluster")["GDP"].mean().sort_values()

    cluster_labels = {}
    labels = ["Low Income", "Middle Income", "High Income"]

    for i, cluster_id in enumerate(cluster_means.index):
        if i < len(labels):
            cluster_labels[cluster_id] = labels[i]
        else:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"

    df["Cluster Name"] = df["Cluster"].map(cluster_labels)

    # =========================
    # FILTER DATA
    # =========================
    if selected_country != "All Countries":
        df_filtered = df[df["Country"] == selected_country]
        df_clean_filtered = df_clean.loc[df_filtered.index]
        clusters_filtered = df_filtered["Cluster"]
    else:
        df_filtered = df
        df_clean_filtered = df_clean
        clusters_filtered = clusters

    # =========================
    # HEADER
    # =========================
    if selected_country != "All Countries":
        st.markdown(f"## {get_flag(selected_country)} {selected_country}")
    else:
        st.markdown("## 🌍 All Countries Overview")

    # =========================
    # OVERVIEW
    # =========================
    if menu == "Overview & EDA":

        col1, col2, col3 = st.columns(3)

        if selected_country != "All Countries":
            col1.metric("🌍 Country", selected_country)
            col3.metric("🧠 Cluster", df_filtered["Cluster Name"].iloc[0])
        else:
            col1.metric("🌍 Countries", df_filtered["Country"].nunique())
            col3.metric("🧠 Total Clusters", df["Cluster"].nunique())

        col2.metric("📊 Features", df_clean_filtered.shape[1])

        st.dataframe(df_filtered.head())
        st.bar_chart(df_clean_filtered.isnull().sum())

        corr = df_clean_filtered.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(corr)
        plt.colorbar(im)
        st.pyplot(fig)

    # =========================
    # FEATURE ANALYSIS
    # =========================
    elif menu == "Feature Analysis":

        feature = st.selectbox("Select Feature", df_clean_filtered.columns)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", round(df_clean_filtered[feature].mean(), 2))
        col2.metric("Max", round(df_clean_filtered[feature].max(), 2))
        col3.metric("Min", round(df_clean_filtered[feature].min(), 2))

        fig, ax = plt.subplots()
        ax.hist(df_clean_filtered[feature], bins=30)
        st.pyplot(fig)

    # =========================
    # CLUSTERING
    # =========================
    elif menu == "Clustering Models":

        st.bar_chart(pd.Series(clusters_filtered).value_counts())

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
        st.pyplot(fig)

        st.dataframe(df_filtered.head())

    # =========================
    # MODEL COMPARISON
    # =========================
    elif menu == "Model Comparison":

        cluster_data = df_clean_filtered.copy()
        cluster_data["Cluster"] = clusters_filtered

        st.dataframe(cluster_data.groupby("Cluster").mean())

    # =========================
    # COUNTRY EXPLORER
    # =========================
    elif menu == "Country Explorer":

        if selected_country == "All Countries":
            st.warning("Please select a country")
        else:
            row = df_filtered.iloc[0]
            row_clean = df_clean_filtered.iloc[0]

            st.markdown(f"### {get_flag(selected_country)} {selected_country}")
            st.write(row_clean)

else:
    st.info("⬅️ Upload dataset to begin")

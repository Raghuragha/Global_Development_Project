import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pycountry

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

    # =========================
    # LOAD DATA
    # =========================
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "Country" not in df.columns:
        st.error("Dataset must contain 'Country'")
        st.stop()

    country_names = df["Country"]

    # =========================
    # GLOBAL COUNTRY FILTER
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
    # TRANSFORM
    # =========================
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")

    X_scaled = scaler.transform(df_clean)
    X_imputed = imputer.fit_transform(X_scaled)
    X_pca = pca.transform(X_imputed)

    clusters = model.predict(X_pca)
    df["Cluster"] = clusters

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
    # 1. OVERVIEW & EDA
    # =========================
    if menu == "Overview & EDA":

        st.markdown("## 📊 Overview & EDA")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌍 Countries", len(df_filtered))
        col2.metric("📊 Features", df_clean_filtered.shape[1])
        col3.metric("🧠 Clusters", len(set(clusters_filtered)))

        st.markdown("### 🔍 Dataset Preview")
        st.dataframe(df_filtered.head())

        st.markdown("### ⚠️ Missing Values")
        st.bar_chart(df_clean_filtered.isnull().sum())

        st.markdown("### 🔥 Correlation Heatmap")
        corr = df_clean_filtered.corr()

        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(corr)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

        plt.colorbar(im)
        st.pyplot(fig)

    # =========================
    # 2. FEATURE ANALYSIS
    # =========================
    elif menu == "Feature Analysis":

        st.markdown("## 🔬 Feature Analysis")

        feature = st.selectbox("Select Feature", df_clean_filtered.columns)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", round(df_clean_filtered[feature].mean(), 2))
        col2.metric("Max", round(df_clean_filtered[feature].max(), 2))
        col3.metric("Min", round(df_clean_filtered[feature].min(), 2))

        fig, ax = plt.subplots()
        ax.hist(df_clean_filtered[feature].dropna(), bins=30)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.boxplot(df_clean_filtered[feature].dropna())
        st.pyplot(fig)

        top = df_filtered.sort_values(by=feature, ascending=False)[["Country", feature]].head(5)
        bottom = df_filtered.sort_values(by=feature, ascending=True)[["Country", feature]].head(5)

        col1, col2 = st.columns(2)
        col1.write("Top 5")
        col1.dataframe(top)

        col2.write("Bottom 5")
        col2.dataframe(bottom)

    # =========================
    # 3. CLUSTERING MODELS
    # =========================
    elif menu == "Clustering Models":

        st.markdown("## 🤖 Clustering Models")

        st.bar_chart(pd.Series(clusters_filtered).value_counts())

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
        st.pyplot(fig)

        st.dataframe(df_filtered.head(20))

    # =========================
    # 4. MODEL COMPARISON
    # =========================
    elif menu == "Model Comparison":

        st.markdown("## 📊 Model Comparison")

        cluster_counts = pd.Series(clusters_filtered).value_counts()
        st.dataframe(cluster_counts.reset_index().rename(
            columns={"index": "Cluster", 0: "Count"}
        ))

        cluster_data = df_clean_filtered.copy()
        cluster_data["Cluster"] = clusters_filtered

        cluster_means = cluster_data.groupby("Cluster").mean()
        st.dataframe(cluster_means)

        feature = st.selectbox("Select Feature", df_clean_filtered.columns)

        fig, ax = plt.subplots()
        cluster_means[feature].plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # =========================
    # 5. COUNTRY EXPLORER
    # =========================
    elif menu == "Country Explorer":

        st.markdown("## 🌍 Country Explorer")

        if selected_country == "All Countries":
            st.warning("Please select a country from sidebar")
        else:
            row = df_filtered.iloc[0]
            row_clean = df_clean_filtered.iloc[0]

            st.markdown(f"""
            <div class="card">
                <h3>{get_flag(selected_country)} {selected_country}</h3>
                <p>Cluster: <b>{int(row['Cluster'])}</b></p>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(4)

            for i, col_name in enumerate(df_clean_filtered.columns[:8]):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="card">
                        <p>{col_name}</p>
                        <div class="metric">{round(row_clean[col_name], 2)}</div>
                    </div>
                    """, unsafe_allow_html=True)

            cluster_id = row["Cluster"]

            cluster_data = df_clean.copy()
            cluster_data["Cluster"] = clusters

            cluster_mean = cluster_data[cluster_data["Cluster"] == cluster_id].mean()

            features = df_clean.columns[:5]

            fig, ax = plt.subplots()
            ax.bar(features, row_clean[features], label="Country")
            ax.bar(features, cluster_mean[features], alpha=0.5, label="Cluster Mean")
            plt.xticks(rotation=45)
            plt.legend()

            st.pyplot(fig)

else:
    st.info("⬅️ Upload dataset to begin")

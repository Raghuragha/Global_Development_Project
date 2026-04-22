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
st.set_page_config(page_title="Global Dev Dashboard", layout="wide")

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
# POPULATION BLOCK
# =========================
def show_population_block(df, selected_country):

    if "Population" not in df.columns:
        return

    st.markdown("### 👥 Population Insights")

    if selected_country != "All Countries":

        country_df = df[df["Country"] == selected_country]

        if not country_df.empty:
            pop_val = country_df["Population"].iloc[0]

            col1, col2 = st.columns([1,2])

            col1.metric("Population", f"{int(pop_val):,}")

            if "Year" in df.columns:
                trend = df[df["Country"] == selected_country][["Year", "Population"]]

                if len(trend) > 1:
                    fig, ax = plt.subplots()
                    ax.plot(trend["Year"], trend["Population"])
                    ax.set_title("Population Trend")
                    col2.pyplot(fig)

    else:
        total_pop = df["Population"].sum()
        st.metric("🌍 Total Population", f"{int(total_pop):,}")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌍 Global Dev Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])

menu = st.sidebar.radio("Navigation", [
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

scaler, pca, model, columns = load_models()

# =========================
# MAIN APP
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    if "Country" not in df.columns:
        st.error("Dataset must contain 'Country'")
        st.stop()

    # =========================
    # REMOVE DUPLICATES (LATEST YEAR)
    # =========================
    if "Year" in df.columns:
        df = df.sort_values("Year").drop_duplicates("Country", keep="last")

    # =========================
    # COUNTRY FILTER
    # =========================
    countries = sorted(df["Country"].unique())

    selected_country = st.sidebar.selectbox(
        "Select Country",
        ["All Countries"] + countries,
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
        df_clean[col] = pd.to_numeric(
            df_clean[col].astype(str)
            .str.replace(',', '')
            .str.replace('$', '')
            .str.replace('%', ''),
            errors='coerce'
        )

    # IMPUTATION
    imputer = SimpleImputer(strategy="mean")

    df_clean = pd.DataFrame(
        imputer.fit_transform(df_clean),
        columns=df_clean.columns,
        index=df_clean.index
    )

    # =========================
    # MODEL
    # =========================
    X_scaled = scaler.transform(df_clean)
    X_pca = pca.transform(X_scaled)

    clusters = model.predict(X_pca)
    df["Cluster"] = clusters

    # =========================
    # CLUSTER LABELS
    # =========================
    cluster_data = df_clean.copy()
    cluster_data["Cluster"] = clusters

    if "GDP" in cluster_data.columns:
        cluster_means = cluster_data.groupby("Cluster")["GDP"].mean().sort_values()
    else:
        cluster_means = cluster_data.groupby("Cluster").mean().mean(axis=1).sort_values()

    labels = ["Low Income", "Middle Income", "High Income"]
    cluster_labels = {}

    for i, cluster_id in enumerate(cluster_means.index):
        cluster_labels[cluster_id] = labels[i] if i < len(labels) else f"Cluster {cluster_id}"

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
    st.markdown(f"## {selected_country}")

    # =========================
    # OVERVIEW
    # =========================
    if menu == "Overview & EDA":

        show_population_block(df, selected_country)

        col1, col2, col3 = st.columns(3)

        if selected_country != "All Countries":
            col1.metric("Country", selected_country)
            col3.metric("Cluster", df_filtered["Cluster Name"].iloc[0])
        else:
            col1.metric("Countries", df["Country"].nunique())
            col3.metric("Clusters", df["Cluster"].nunique())

        col2.metric("Features", df_clean_filtered.shape[1])

        st.dataframe(df_filtered.head())

    # =========================
    # FEATURE ANALYSIS
    # =========================
    elif menu == "Feature Analysis":

        show_population_block(df, selected_country)

        feature = st.selectbox("Feature", df_clean_filtered.columns)

        st.metric("Mean", round(df_clean_filtered[feature].mean(), 2))

        fig, ax = plt.subplots()
        ax.hist(df_clean_filtered[feature], bins=30)
        st.pyplot(fig)

    # =========================
    # CLUSTERING
    # =========================
    elif menu == "Clustering Models":

        show_population_block(df, selected_country)

        st.bar_chart(pd.Series(clusters_filtered).value_counts())

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
        st.pyplot(fig)

    # =========================
    # MODEL COMPARISON
    # =========================
    elif menu == "Model Comparison":

        show_population_block(df, selected_country)

        temp = df_clean_filtered.copy()
        temp["Cluster"] = clusters_filtered

        st.dataframe(temp.groupby("Cluster").mean())

    # =========================
    # COUNTRY EXPLORER
    # =========================
    elif menu == "Country Explorer":

        show_population_block(df, selected_country)

        if selected_country != "All Countries":
            st.write(df_clean_filtered.iloc[0])
        else:
            st.warning("Select a country")

else:
    st.info("⬅️ Upload dataset to begin")

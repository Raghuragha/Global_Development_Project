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
# MAIN
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    # =========================
    # FIX MULTIPLE ROWS
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
            df_clean[col].astype(str).str.replace(',', '').str.replace('$', ''),
            errors='coerce'
        )

    # IMPUTE
    imputer = SimpleImputer(strategy="mean")
    df_clean[:] = imputer.fit_transform(df_clean)

    # =========================
    # MODEL
    # =========================
    X_scaled = scaler.transform(df_clean)
    X_pca = pca.transform(X_scaled)

    clusters = model.predict(X_pca)
    df["Cluster"] = clusters

    # =========================
    # FILTER
    # =========================
    if selected_country != "All Countries":
        df_filtered = df[df["Country"] == selected_country]
        df_clean_filtered = df_clean.loc[df_filtered.index]
    else:
        df_filtered = df
        df_clean_filtered = df_clean

    # =========================
    # LAYOUT (LEFT + RIGHT)
    # =========================
    left, right = st.columns([1, 3])

    # =========================
    # LEFT SIDE (POPULATION)
    # =========================
    with left:
        st.markdown("## 👥 Population")

        if "Population" in df.columns:

            if selected_country != "All Countries":
                pop_val = df_filtered["Population"].iloc[0]
                st.metric("Population", f"{int(pop_val):,}")

            # Graph
            if "Year" in df.columns:
                pop_trend = df[df["Country"] == selected_country][["Year", "Population"]]

                fig, ax = plt.subplots()
                ax.plot(pop_trend["Year"], pop_trend["Population"])
                ax.set_title("Population Trend")
                st.pyplot(fig)

        else:
            st.warning("Population column not found")

    # =========================
    # RIGHT SIDE (MAIN CONTENT)
    # =========================
    with right:

        st.markdown(f"## {selected_country}")

        if menu == "Overview & EDA":

            st.metric("Countries", df_filtered["Country"].nunique())
            st.metric("Features", df_clean_filtered.shape[1])

            st.dataframe(df_filtered.head())

        elif menu == "Feature Analysis":

            feature = st.selectbox("Feature", df_clean_filtered.columns)

            st.metric("Mean", round(df_clean_filtered[feature].mean(), 2))

            fig, ax = plt.subplots()
            ax.hist(df_clean_filtered[feature], bins=30)
            st.pyplot(fig)

        elif menu == "Clustering Models":

            st.bar_chart(pd.Series(clusters).value_counts())

        elif menu == "Model Comparison":

            df_clean_filtered["Cluster"] = df_filtered["Cluster"]
            st.dataframe(df_clean_filtered.groupby("Cluster").mean())

        elif menu == "Country Explorer":

            if selected_country != "All Countries":
                st.write(df_clean_filtered.iloc[0])

else:
    st.info("Upload dataset")

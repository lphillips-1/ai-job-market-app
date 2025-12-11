# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
from math import sqrt
# We will use the OpenAI client, which is imported in the helper function below.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(layout="wide", page_title="AI Job Market — Presentation Dashboard (OpenAI Only)")

# ---------------------------
# Helpers
# ---------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def extract_top_skills(df: pd.DataFrame, col="required_skills", top_n=20):
    if col not in df.columns:
        return []
    s = df[col].fillna("").astype(str).str.lower()
    exploded = s.str.split(r'\s*,\s*').explode().str.strip()
    exploded = exploded[exploded != ""]
    top = exploded.value_counts().head(top_n).index.tolist()
    return top

def make_skill_features(df: pd.DataFrame, top_skills, col="required_skills"):
    s = df[col].fillna("").astype(str).str.lower() if col in df.columns else pd.Series([""]*len(df))
    skill_df = pd.DataFrame(index=df.index)
    for skill in top_skills:
        # defensive check for commas/spaces
        skill_df[f"skill__{skill}"] = s.apply(lambda x: 1 if skill in [k.strip() for k in x.split(",") if k.strip()!=''] else 0)
    return skill_df

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def safe_feature_names_from_preprocessor(preprocessor: ColumnTransformer, categorical_cols, numeric_cols):
    """
    Best-effort to produce feature names after transforming with ColumnTransformer.
    If we cannot, return concatenation of categorical_ohe_names + numeric_cols as fallback.
    """
    feature_names = []
    # categorical
    try:
        ohe = preprocessor.named_transformers_.get('cat', None)
        if ohe is not None and hasattr(ohe, 'get_feature_names_out'):
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist() 
        else:
            cat_names = [f"cat__{c}" for c in categorical_cols]
    except Exception:
        cat_names = [f"cat__{c}" for c in categorical_cols]
    feature_names.extend(cat_names)
    # numeric names
    feature_names.extend(numeric_cols)
    return feature_names

# ---------------------------
# Sidebar controls (MODIFIED)
# ---------------------------
st.sidebar.title("Controls & Settings")
st.sidebar.markdown("**OpenAI API Key (sk-...)**")

# API Key input is now purely for OpenAI
API_KEY_INPUT = st.sidebar.text_input("Paste your OpenAI API key", type="password", key="openai_key")
BASE_URL_USED = "https://api.openai.com/v1" # Standard OpenAI endpoint

# ---------------------------
# Streamlit State Variables for Chat Client
# ---------------------------
if 'llm_client' not in st.session_state:
    st.session_state['llm_client'] = None
    
# Function to safely initialize the client
def get_llm_client(api_key, base_url):
    # Import OpenAI here to ensure it is only accessed if needed
    from openai import OpenAI 
    if not api_key:
        return None
    try:
        # Client initialized only with standard OpenAI API settings
        client = OpenAI(
            api_key=api_key,
            base_url=base_url 
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

st.sidebar.markdown("---")
st.sidebar.write("Tip: Upload dataset in the 'Upload & Overview' tab first.")
st.sidebar.markdown("---")
st.sidebar.header("Model & Clustering")
train_models_btn = st.sidebar.button("Train Models")
n_clusters_sidebar = st.sidebar.slider("Specify the number of clusters to group variations in median salary and typical experience/skill patterns.", 2, 8, 4)
top_skills_n_sidebar = st.sidebar.slider("Top skills to extract", 5, 50, 20)


# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_data
def load_csv_bytes(uploaded):
    return pd.read_csv(uploaded)

@st.cache_data
def load_local_csv(path="ai_job_dataset.csv"):
    return pd.read_csv(path)

# ---------------------------
# Tabs (MODIFIED tab names)
# ---------------------------
tabs = st.tabs(["Upload & Overview", "Train Models", "Make Predictions", "Explore Clusters", "Chat with OpenAI"])
tab_overview, tab_train, tab_predict, tab_cluster, tab_chat = tabs

# ---------------------------
# Session-state defaults
# ---------------------------
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_proc' not in st.session_state:
    st.session_state['df_proc'] = None
if 'top_skills' not in st.session_state:
    st.session_state['top_skills'] = []
if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False

# ---------------------------
# TAB 1 — Upload & Overview
# ---------------------------
with tab_overview:
    st.header("Part 1 — Upload Market Overview")
    st.write("Upload the dataset (`ai_job_dataset.csv`) or load the sample file from the working directory.")
    col_u1, col_u2 = st.columns([3,1])
    with col_u1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            st.session_state['df_raw'] = load_csv_bytes(uploaded_file)
            st.success("File uploaded and loaded.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    df_raw = st.session_state['df_raw']
    if df_raw is None:
        st.info("Please upload the dataset to begin. The other tabs will become active.")

    if df_raw is not None:
        # basic cleaning & prepare display
        df_raw = clean_cols(df_raw)
        st.session_state['df_raw'] = df_raw

        st.subheader("Market at a glance")
        df_disp = df_raw.copy()
        df_disp = ensure_numeric(df_disp, ['salary_usd', 'remote_ratio', 'years_experience', 'benefits_score', 'job_description_length'])

        c1, c2, c3 = st.columns(3)

        # Salary histogram
        with c1:
            st.markdown("**Salary distribution (USD)**")
            if 'salary_usd' in df_disp.columns and df_disp['salary_usd'].dropna().shape[0] > 0:
                fig1, ax1 = plt.subplots(figsize=(4,2.2))
                ax1.hist(df_disp['salary_usd'].dropna(), bins=30)
                ax1.set_xlabel("USD")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)
                st.caption(f"Median salary: ${int(df_disp['salary_usd'].median()):,}")
            else:
                st.info("No numeric salary data available.")

        # Geography bar
        with c2:
            st.markdown("**Geographic spread (company_location)**")
            if 'company_location' in df_disp.columns:
                top_loc = df_disp['company_location'].value_counts().head(10)
                fig2, ax2 = plt.subplots(figsize=(4,2.2))
                top_loc.plot(kind='barh', ax=ax2)
                ax2.invert_yaxis()
                st.pyplot(fig2)
            else:
                st.info("No company_location column found.")

        # Top job titles
        with c3:
            st.markdown("**Most common job titles**")
            if 'job_title' in df_disp.columns:
                top_titles = df_disp['job_title'].value_counts().head(10)
                fig3, ax3 = plt.subplots(figsize=(4,2.2))
                top_titles.plot(kind='bar', ax=ax3)
                ax3.set_xlabel("Count")
                st.pyplot(fig3)
            else:
                st.info("No job_title column found.")

        st.markdown("---")
        st.subheader("Dataset preview & quick stats")
        st.write(f"Rows: {len(df_disp):,} — Columns: {df_disp.shape[1]}")
        with st.expander("Show raw data (first 200 rows)"):
            st.dataframe(df_disp.head(200))

        # Prepare processed dataframe for modeling
        st.markdown("Preparing dataset for modeling (automated)...")
        df_proc = df_disp.copy()

        # simple fills
        if 'company_size' in df_proc.columns:
            df_proc['company_size'] = df_proc['company_size'].fillna('M').astype(str)
        if 'experience_level' in df_proc.columns:
            df_proc['experience_level'] = df_proc['experience_level'].fillna('EN').astype(str)
        for n in ['salary_usd', 'remote_ratio', 'years_experience', 'job_description_length', 'benefits_score']:
            if n in df_proc.columns:
                df_proc[n] = pd.to_numeric(df_proc[n], errors='coerce').fillna(df_proc[n].median())

        # top skills extraction
        top_skills = extract_top_skills(df_proc, col="required_skills", top_n=top_skills_n_sidebar)
        skills_df = make_skill_features(df_proc, top_skills, col="required_skills")
        if skills_df.shape[1] > 0:
            df_proc = pd.concat([df_proc.reset_index(drop=True), skills_df.reset_index(drop=True)], axis=1)

        st.session_state['df_proc'] = df_proc
        st.session_state['top_skills'] = top_skills

        st.success("Dataset prepared. Move to 'Train Models' to continue.")

# ---------------------------
# TAB 2 — Train Models
# ---------------------------
with tab_train:
    st.header("Part 2 — Train Models")
    st.write("Press **Train Models** in the sidebar to train all three models. Metrics will be shown using a held-out test split.")

    if st.session_state['df_proc'] is None:
        st.info("Upload data first in 'Upload & Overview'.")

    if st.session_state['df_proc'] is not None:
        df_proc = st.session_state['df_proc']
        top_skills = st.session_state['top_skills']

        # Drop columns not useful for modeling
        drop_cols = ['job_id', 'job_title', 'salary_currency', 'employee_residence',
                    'posting_date', 'application_deadline', 'required_skills',
                    'job_description', 'company_name']
        drop_cols = [c for c in drop_cols if c in df_proc.columns]
        data_for_model = df_proc.drop(columns=drop_cols, errors='ignore').copy()

        # Ensure targets present
        missing_targets = [t for t in ['salary_usd', 'remote_ratio', 'company_size'] if t not in data_for_model.columns]
        if missing_targets:
            st.error(f"Missing required target column(s): {', '.join(missing_targets)}")

        if not missing_targets:
            y_salary = data_for_model['salary_usd'].astype(float)
            y_remote = data_for_model['remote_ratio'].astype(float)
            y_size = data_for_model['company_size'].astype(str)
            X_all = data_for_model.drop(columns=['salary_usd', 'remote_ratio', 'company_size'])

            numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_all.select_dtypes(include=['object', 'category']).columns.tolist()

            # ColumnTransformer: OneHotEncoder (sparse_output for newer sklearn), StandardScaler for numeric
            preprocessor = ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ], remainder='drop')

            if train_models_btn:
                st.info("Training models (showing evaluation on a held-out test split)...")
                with st.spinner("Training..."):
                    # train/test split for evaluation
                    train_idx, test_idx = train_test_split(np.arange(len(X_all)), test_size=0.2, random_state=42)
                    X_train = X_all.iloc[train_idx]
                    X_test = X_all.iloc[test_idx]

                    # Salary model
                    salary_pipe = Pipeline(steps=[
                        ('pre', preprocessor),
                        ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
                    ])
                    salary_pipe.fit(X_train, y_salary.iloc[train_idx])
                    y_pred_sal = salary_pipe.predict(X_test)
                    mae_sal = mean_absolute_error(y_salary.iloc[test_idx], y_pred_sal)
                    mse_sal = mean_squared_error(y_salary.iloc[test_idx], y_pred_sal)
                    rmse_sal = sqrt(mse_sal)
                    r2_sal = r2_score(y_salary.iloc[test_idx], y_pred_sal)

                    # Remote model
                    remote_pipe = Pipeline(steps=[
                        ('pre', preprocessor),
                        ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
                    ])
                    remote_pipe.fit(X_train, y_remote.iloc[train_idx])
                    y_pred_rem = remote_pipe.predict(X_test)
                    mae_rem = mean_absolute_error(y_remote.iloc[test_idx], y_pred_rem)
                    mse_rem = mean_squared_error(y_remote.iloc[test_idx], y_pred_rem)
                    rmse_rem = sqrt(mse_rem)
                    r2_rem = r2_score(y_remote.iloc[test_idx], y_pred_rem)

                    # Company size classifier
                    size_pipe = Pipeline(steps=[
                        ('pre', preprocessor),
                        ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
                    ])
                    size_pipe.fit(X_train, y_size.iloc[train_idx])
                    y_pred_size = size_pipe.predict(X_test)
                    acc_size = accuracy_score(y_size.iloc[test_idx], y_pred_size)

                    # Display metrics
                    st.subheader("Model performance (test set)")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Salary (Regression)**")
                        st.metric("MAE", f"${mae_sal:,.0f}")
                        st.metric("RMSE", f"${rmse_sal:,.0f}")
                        st.metric("R²", f"{r2_sal:.3f}")
                    with c2:
                        st.markdown("**Remote Ratio (Regression)**")
                        st.metric("MAE", f"{mae_rem:.2f}")
                        st.metric("RMSE", f"{rmse_rem:.2f}")
                        st.metric("R²", f"{r2_rem:.3f}")
                    with c3:
                        st.markdown("**Company Size (Classification)**")
                        st.metric("Accuracy", f"{acc_size:.3f}")

                    # Feature importances for salary (best-effort)
                    try:
                        feature_names = safe_feature_names_from_preprocessor(preprocessor, categorical_cols, numeric_cols)
                        importances = salary_pipe.named_steps['model'].feature_importances_
                        if len(importances) == len(feature_names):
                            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(20)
                            fig_imp, ax_imp = plt.subplots(figsize=(6,4))
                            ax_imp.barh(imp_df['feature'][::-1], imp_df['importance'][::-1])
                            ax_imp.set_title("Top 20 Feature Importances (Salary)")
                            st.pyplot(fig_imp)
                        else:
                            st.info("Feature importance not shown due to mismatch between feature names and importances.")
                    except Exception as e:
                        st.info(f"Feature importance extraction failed (skipped). Error: {e}")


                    # Retrain final models on FULL dataset for prediction
                    salary_full = Pipeline(steps=[('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
                    remote_full = Pipeline(steps=[('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
                    size_full = Pipeline(steps=[('pre', preprocessor), ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])

                    salary_full.fit(X_all, y_salary)
                    remote_full.fit(X_all, y_remote)
                    size_full.fit(X_all, y_size)

                    # Save in session state
                    st.session_state['model_salary'] = salary_full
                    st.session_state['model_remote'] = remote_full
                    st.session_state['model_size'] = size_full
                    st.session_state['X_columns'] = X_all.columns.tolist()
                    st.session_state['preprocessor'] = preprocessor
                    st.session_state['models_trained'] = True

                    st.success("Models trained on full dataset and ready for predictions.")
            else:
                if st.session_state['models_trained']:
                    st.success("Models are already trained in this session.")
                else:
                    st.info("Click 'Train Models' in sidebar to train models on the uploaded dataset.")

# ---------------------------
# TAB 3 — Make Predictions
# ---------------------------
with tab_predict:
    st.header("Part 3 — Make Predictions (45 sec)")
    st.write("Build a candidate profile (e.g., Sarah) and click Predict to get data-driven outputs.")

    if not st.session_state.get('models_trained', False):
        st.info("Train the models first (sidebar button) or go to 'Train Models' tab.")
    
    if st.session_state.get('models_trained', False):
        model_salary = st.session_state['model_salary']
        model_remote = st.session_state['model_remote']
        model_size = st.session_state['model_size']
        X_columns = st.session_state['X_columns']
        df_proc = st.session_state['df_proc']
        top_skills = st.session_state['top_skills']

        # Profile builder UI
        st.subheader("Profile builder")
        col1, col2, col3 = st.columns(3)
        with col1:
            title_input = st.text_input("Job title (free text)", value="Data Scientist")
            exp_input = st.selectbox("Experience level",
                                    options=sorted(df_proc['experience_level'].unique().tolist()) if 'experience_level' in df_proc.columns else ['EN','SE','MI'])
            years_input = st.number_input("Years experience", min_value=0, max_value=40, value=int(df_proc['years_experience'].median() if 'years_experience' in df_proc.columns else 3))
        with col2:
            loc_input = st.selectbox("Company location",
                                    options=sorted(df_proc['company_location'].dropna().unique().tolist()) if 'company_location' in df_proc.columns else ['USA'])
            size_input = st.selectbox("Company size",
                                        options=sorted(df_proc['company_size'].dropna().unique().tolist()) if 'company_size' in df_proc.columns else ['M','S','L'])
            remote_pref = st.slider("Preferred remote ratio (0-100)", 0, 100, 50)
        with col3:
            selected_skills = st.multiselect("Select skills (top extracted)", options=top_skills[:30], default=top_skills[:3] if len(top_skills)>=3 else [])
            job_desc_len = st.number_input("Job description length (chars)", value=int(df_proc['job_description_length'].median() if 'job_description_length' in df_proc.columns else 1000))

        if st.button("Predict for this profile"):
            # Build row with default values then overwrite with inputs
            row = pd.DataFrame(index=[0], columns=X_columns)
            # default numeric fills
            for col in X_columns:
                if col in df_proc.columns:
                    if pd.api.types.is_numeric_dtype(df_proc[col]):
                        row.at[0, col] = float(df_proc[col].median())
                    else:
                        row.at[0, col] = ""
                else:
                    # if the column wasn't in df_proc (rare), set empty numeric default
                    row.at[0, col] = 0.0 if col in [] else ""

            # fill with our inputs
            if 'experience_level' in row.columns:
                row.at[0, 'experience_level'] = exp_input
            if 'years_experience' in row.columns:
                row.at[0, 'years_experience'] = years_input
            if 'company_location' in row.columns:
                row.at[0, 'company_location'] = loc_input
            if 'company_size' in row.columns:
                row.at[0, 'company_size'] = size_input
            if 'remote_ratio' in row.columns:
                row.at[0, 'remote_ratio'] = remote_pref
            if 'job_description_length' in row.columns:
                row.at[0, 'job_description_length'] = job_desc_len

            # skill columns
            for s in top_skills:
                colname = f"skill__{s}"
                if colname in row.columns:
                    row.at[0, colname] = 1 if s in selected_skills else 0

            # approximate job_title handling (if exists among categorical cols)
            if 'job_title' in row.columns:
                row.at[0, 'job_title'] = title_input

            # convert numeric columns safely
            for c in row.columns:
                if c in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[c]):
                    row[c] = pd.to_numeric(row[c], errors='coerce').fillna(df_proc[c].median())

            # ensure order
            row = row[X_columns]

            # Make predictions
            pred_salary = float(model_salary.predict(row)[0])
            pred_remote = float(model_remote.predict(row)[0])
            pred_size = model_size.predict(row)[0]

            # Show results prominently
            r1, r2, r3 = st.columns([2,1,1])
            with r1:
                st.markdown("### Predicted Salary (USD)")
                st.markdown(f"<h1 style='color:#0b4c8a'>${pred_salary:,.0f}</h1>", unsafe_allow_html=True)
                st.caption("Model prediction trained on the full dataset")
            with r2:
                st.markdown("### Predicted Remote Ratio")
                st.metric("", f"{pred_remote:.0f}%")
            with r3:
                st.markdown("### Predicted Company Size")
                st.markdown(f"**{pred_size}**")
            st.markdown("---")
            st.info("These predictions are data-driven outputs from Random Forest models trained on your dataset.")

# ---------------------------
# TAB 4 — Explore Clusters
# ---------------------------
with tab_cluster:
    st.header("Part 4 — Explore Clusters")
    st.write("K-Means clusters (each dot is a job). Adjust k to specify the number of clusters to group variations in median salary and typical experience/skill patterns.")

    if st.session_state['df_proc'] is None:
        st.info("Upload dataset first.")

    if st.session_state['df_proc'] is not None:
        df_proc = st.session_state['df_proc']
        top_skills = st.session_state['top_skills']
        k = st.slider("Select k (clusters)", 2, 8, n_clusters_sidebar)

        # Feature selection for clustering
        cluster_numeric = [c for c in ['salary_usd', 'years_experience', 'remote_ratio', 'benefits_score', 'job_description_length'] if c in df_proc.columns]
        skill_subset = [f"skill__{s}" for s in top_skills[:10] if f"skill__{s}" in df_proc.columns]
        cluster_features = cluster_numeric + skill_subset

        if len(cluster_features) < 2:
            st.error("Not enough numeric features for clustering (need salary/years/skills).")

        if len(cluster_features) >= 2:
            from sklearn.preprocessing import StandardScaler
            cluster_df = df_proc[cluster_features].fillna(0).copy()
            scaler = StandardScaler()
            cluster_scaled = scaler.fit_transform(cluster_df)

            # PCA for 2D plot
            pca = PCA(n_components=2, random_state=42)
            components = pca.fit_transform(cluster_scaled)

            # KMeans
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(cluster_scaled)

            fig, ax = plt.subplots(figsize=(8,5))
            scatter = ax.scatter(components[:,0], components[:,1], c=labels, cmap='tab10', alpha=0.75, s=35)
            ax.set_title("K-Means clusters (PCA 2D projection)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)

            # Build cluster summaries robustly and cast to strings before display
            summary = []
            for cluster_id in range(k):
                idxs = np.where(labels == cluster_id)[0]
                part = df_proc.iloc[idxs]
                nrows = int(len(part))
                avg_salary = float(part['salary_usd'].mean()) if 'salary_usd' in part.columns and not part['salary_usd'].isna().all() else np.nan
                avg_exp = float(part['years_experience'].mean()) if 'years_experience' in part.columns and not part['years_experience'].isna().all() else np.nan
                top_titles = part['job_title'].value_counts().head(3).index.tolist() if 'job_title' in part.columns else []
                # top skills percentages
                skills_info = []
                for s in top_skills[:10]:
                    coln = f"skill__{s}"
                    if coln in part.columns:
                        pct = part[coln].mean()
                        if pd.notna(pct) and pct > 0:
                            skills_info.append((s, round(float(pct), 2)))
                summary.append({
                    "cluster": int(cluster_id),
                    "n_jobs": nrows,
                    "avg_salary": round(avg_salary, 1) if not np.isnan(avg_salary) else None,
                    "avg_experience": round(avg_exp, 1) if not np.isnan(avg_exp) else None,
                    "top_titles": top_titles,
                    "top_skills": skills_info[:5]
                })

            # Convert to DataFrame then cast all cells to strings to avoid PyArrow conversions
            summary_df = pd.DataFrame(summary)
            # Replace NaN with None first (optional), then cast
            summary_df = summary_df.where(pd.notnull(summary_df), None).astype(str)
            st.write(summary_df)

            st.caption("Interpret clusters by median salary and typical experience/skills.")

# ---------------------------
# TAB 5 — Chat with OpenAI (MODIFIED)
# ---------------------------
with tab_chat:
    st.header("Chat with an AI Advisor (OpenAI)")
    st.write("Ask industry questions, research questions, or interpretation questions.")

    if not API_KEY_INPUT:
        st.warning("Please enter your OpenAI API key in the sidebar to enable chat.")
    else:
        # Initialize client using the API Key and standard OpenAI endpoint
        st.session_state['llm_client'] = get_llm_client(API_KEY_INPUT, BASE_URL_USED)
        client = st.session_state['llm_client']

        if client is not None:
            if "messages" not in st.session_state:
                # Initialize chat history with a system message
                st.session_state.messages = [
                    {"role": "system", "content": "You are a helpful AI advisor, providing data-driven insights and general knowledge."}
                ]

            # Display chat history
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    continue
                role = "assistant" if msg["role"] == "assistant" else "user"
                st.chat_message(role).write(msg["content"])

            user_input = st.chat_input("Ask something…")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini", # Using the model from your original request
                        messages=st.session_state.messages
                    )
                    reply = completion.choices[0].message.content 
                except Exception as e:
                    # If API call fails, remove the user's message from history
                    st.session_state.messages.pop() 
                    st.error(f"API call failed. Check your key, ensure the model 'gpt-4o-mini' is accessible, and verify billing information. Error: {e}")
                    reply = None

                if reply:
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.chat_message("assistant").write(reply)
        else:
            st.error("Failed to set up the OpenAI client. Please ensure your API key is correct and valid.")

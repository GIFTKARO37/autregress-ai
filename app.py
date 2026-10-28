import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import io

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="AutoRegress AI", page_icon="🤖", layout="wide")

st.title("🤖 AutoRegress AI")
st.caption("Automated Machine Learning | Regression Platform")

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "best_model" not in st.session_state:
    st.session_state.best_model = None

if "features" not in st.session_state:
    st.session_state.features = []

if "target" not in st.session_state:
    st.session_state.target = None

if "results_df" not in st.session_state:
    st.session_state.results_df = None

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Dataset Analysis", "Regression Lab", "Prediction Center"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = uploaded_file.name
    except Exception as e:
        st.error(f"File reading error: {e}")

df = st.session_state.get("dataset")

if page == "Dashboard":
    st.header("📊 Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("ML Algorithms", "4")
    c2.metric("Dataset", "Uploaded" if df is not None else "Not uploaded")
    c3.metric("Model Status", "Trained" if st.session_state.best_model is not None else "Not trained")

    if df is not None:
        st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Upload a CSV or Excel dataset using the sidebar.")

elif page == "Dataset Analysis":
    st.header("📁 Dataset Analysis")
    if df is None:
        st.warning("Upload a dataset from the sidebar first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Missing Values")
        st.dataframe(df.isna().sum().to_frame("Missing Values"))

        st.subheader("Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            col = st.selectbox("Plot numeric column", numeric)
            st.plotly_chart(px.histogram(df, x=col, title=f"Distribution of {col}"), use_container_width=True)

elif page == "Regression Lab":
    st.header("🧪 Regression Lab")

    if df is None:
        st.warning("Upload a dataset from the sidebar first.")
    elif len(df) < 5:
        st.error("Dataset needs at least 5 rows.")
    else:
        target = st.selectbox("Select target column to predict", df.columns.tolist())
        available = [c for c in df.columns if c != target]
        features = st.multiselect("Select input features", available, default=available)
        test_percent = st.slider("Test data percentage", 10, 40, 20, 5)

        if st.button("🚀 Train All 4 Models", type="primary"):
            if not features:
                st.error("Select at least one feature.")
            else:
                try:
                    work = df[features + [target]].copy()
                    work[target] = pd.to_numeric(work[target], errors="coerce")
                    work = work.replace([np.inf, -np.inf], np.nan)
                    work = work.dropna(subset=[target])

                    if len(work) < 5:
                        st.error("Not enough valid target values.")
                        st.stop()

                    X = work[features]
                    y = work[target]

                    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                    categorical_features = [c for c in features if c not in numeric_features]

                    transformers = []
                    if numeric_features:
                        transformers.append((
                            "num",
                            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                            numeric_features
                        ))
                    if categorical_features:
                        transformers.append((
                            "cat",
                            Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore"))
                            ]),
                            categorical_features
                        ))

                    preprocess = ColumnTransformer(transformers=transformers)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_percent / 100, random_state=42
                    )

                    models = {
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
                    }

                    trained = {}
                    predictions = {}
                    rows = []
                    progress = st.progress(0)

                    for i, (name, model) in enumerate(models.items()):
                        pipeline = Pipeline([
                            ("preprocess", preprocess),
                            ("model", model)
                        ])
                        pipeline.fit(X_train, y_train)
                        pred = pipeline.predict(X_test)

                        trained[name] = pipeline
                        predictions[name] = pred
                        rows.append({
                            "Model": name,
                            "MAE": mean_absolute_error(y_test, pred),
                            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                            "R² Score": r2_score(y_test, pred) if len(y_test) > 1 else np.nan
                        })
                        progress.progress((i + 1) / len(models))

                    progress.empty()
                    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
                    best_name = results.iloc[0]["Model"]

                    st.session_state.trained_models = trained
                    st.session_state.best_model = trained[best_name]
                    st.session_state.best_name = best_name
                    st.session_state.features = features
                    st.session_state.target = target
                    st.session_state.results_df = results
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.best_pred = predictions[best_name]

                    st.success(f"Training complete! Best model: {best_name}")
                    st.subheader("Model Comparison")
                    st.dataframe(results.style.format({
                        "MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.4f}"
                    }), use_container_width=True)

                    chart = pd.DataFrame({
                        "Actual": np.asarray(y_test),
                        "Predicted": np.asarray(predictions[best_name])
                    })
                    st.plotly_chart(
                        px.scatter(chart, x="Actual", y="Predicted", title="Actual vs Predicted"),
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Metrics CSV",
                        results.to_csv(index=False).encode("utf-8"),
                        "model_metrics.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Training error: {e}")

        if st.session_state.results_df is not None:
            st.subheader("Latest Model Results")
            st.dataframe(st.session_state.results_df, use_container_width=True)

elif page == "Prediction Center":
    st.header("🔮 Prediction Center")

    model = st.session_state.best_model
    features = st.session_state.features
    target = st.session_state.target

    if model is None or not features or not target:
        st.info("Go to Regression Lab, select a target column and train the models first.")
    else:
        st.success(f"Active model: {st.session_state.get('best_name', 'Best model')}")
        st.write(f"**Target being predicted: {target}**")

        values = {}
        for feature in features:
            if df is not None and feature in df.columns:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    median = pd.to_numeric(df[feature], errors="coerce").median()
                    if pd.isna(median):
                        median = 0.0
                    values[feature] = st.number_input(
                        feature, value=float(median), key=f"predict_{feature}"
                    )
                else:
                    options = df[feature].dropna().astype(str).unique().tolist()
                    if options:
                        values[feature] = st.selectbox(
                            feature, options, key=f"predict_{feature}"
                        )
                    else:
                        values[feature] = st.text_input(feature, key=f"predict_{feature}")
            else:
                values[feature] = st.text_input(feature, key=f"predict_{feature}")

        if st.button("🔍 Predict", type="primary"):
            try:
                input_df = pd.DataFrame([values], columns=features)
                prediction = float(model.predict(input_df)[0])
                st.metric(f"Predicted {target}", f"{prediction:,.4f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.divider()
        st.subheader("📥 Download Best Trained Model")
        st.warning("Only load model files you created or trust. Pickle files can execute code when loaded.")
        model_bytes = pickle.dumps(model)
        st.download_button(
            "Download Model (.pkl)",
            data=model_bytes,
            file_name="autregress_best_model.pkl",
            mime="application/octet-stream"
        )

        if df is not None and "X_test" in st.session_state:
            st.subheader("⭐ Feature Importance")
            try:
                importance = permutation_importance(
                    model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    n_repeats=5,
                    random_state=42,
                    scoring="neg_mean_absolute_error"
                )
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance.importances_mean
                }).sort_values("Importance", ascending=True)

                st.plotly_chart(
                    px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                           title="Permutation Feature Importance"),
                    use_container_width=True
                )
            except Exception as e:
                st.info(f"Feature importance unavailable for this dataset: {e}")

st.sidebar.divider()
st.sidebar.caption("AutoRegress AI | ML Platform")
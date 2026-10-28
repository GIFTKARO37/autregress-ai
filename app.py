import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(
    page_title="AutoRegress AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoRegress AI")
st.subheader("Advanced Machine Learning Platform")
st.write("Upload your dataset and train regression models automatically.")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Dataset Analysis", "Regression Lab", "Prediction Center"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")

if page == "Dashboard":
    st.title("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Algorithms", "4")
    c2.metric("Dataset Status", "Uploaded" if df is not None else "Not uploaded")
    c3.metric("Model Status", "Ready" if df is not None else "Waiting for data")

    st.info("Upload a CSV or Excel dataset from the sidebar to begin.")

    if df is not None:
        st.success("Dataset uploaded successfully!")
        st.write("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)

elif page == "Dataset Analysis":
    st.title("📁 Dataset Analysis")

    if df is None:
        st.warning("Please upload a CSV or Excel file from the sidebar.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Missing Values by Column")
        missing = df.isna().sum()
        st.dataframe(
            missing[missing > 0].rename("Missing Values").to_frame(),
            use_container_width=True
        )

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.subheader("Numeric Column Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox("Select column", numeric_cols)
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

elif page == "Regression Lab":
    st.title("🧪 Regression Lab")

    if df is None:
        st.warning("Upload a dataset from the sidebar first.")
    elif len(df) < 5:
        st.error("Dataset needs at least 5 rows for training.")
    else:
        st.write("Choose the value you want the AI to predict.")

        target = st.selectbox("Select target column", df.columns.tolist())

        available_features = [col for col in df.columns if col != target]
        features = st.multiselect(
            "Select input features",
            available_features,
            default=available_features
        )

        test_size = st.slider(
            "Test data percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )

        if st.button("🚀 Train All 4 Models", type="primary"):
            if not features:
                st.error("Please select at least one input feature.")
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

                    numeric_features = X.select_dtypes(
                        include=np.number
                    ).columns.tolist()

                    categorical_features = [
                        col for col in features if col not in numeric_features
                    ]

                    transformers = []

                    if numeric_features:
                        numeric_pipeline = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())
                        ])
                        transformers.append(
                            ("numeric", numeric_pipeline, numeric_features)
                        )

                    if categorical_features:
                        categorical_pipeline = Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ])
                        transformers.append(
                            ("categorical", categorical_pipeline, categorical_features)
                        )

                    preprocessing = ColumnTransformer(transformers=transformers)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size / 100,
                        random_state=42
                    )

                    models = {
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(
                            random_state=42, max_depth=10
                        ),
                        "Random Forest": RandomForestRegressor(
                            n_estimators=100, random_state=42
                        ),
                        "Gradient Boosting": GradientBoostingRegressor(
                            random_state=42
                        )
                    }

                    results = []
                    trained_models = {}
                    predictions = {}

                    progress = st.progress(0)
                    status = st.empty()

                    for i, (name, model) in enumerate(models.items()):
                        status.write(f"Training {name}...")

                        pipeline = Pipeline([
                            ("preprocessing", preprocessing),
                            ("model", model)
                        ])

                        pipeline.fit(X_train, y_train)
                        pred = pipeline.predict(X_test)

                        mae = mean_absolute_error(y_test, pred)
                        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                        r2 = r2_score(y_test, pred) if len(y_test) > 1 else np.nan

                        results.append({
                            "Model": name,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R² Score": r2
                        })

                        trained_models[name] = pipeline
                        predictions[name] = pred
                        progress.progress((i + 1) / len(models))

                    status.empty()
                    progress.empty()

                    results_df = pd.DataFrame(results).sort_values(
                        "RMSE", ascending=True
                    ).reset_index(drop=True)

                    best_name = results_df.loc[0, "Model"]
                    best_model = trained_models[best_name]
                    best_pred = predictions[best_name]

                    st.session_state["trained_models"] = trained_models
                    st.session_state["best_model"] = best_model
                    st.session_state["best_name"] = best_name
                    st.session_state["features"] = features
                    st.session_state["target"] = target
                    st.session_state["results_df"] = results_df
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test
                    st.session_state["best_pred"] = best_pred

                    st.success("All 4 models trained successfully!")

                    st.subheader("🏆 Best Model")
                    st.success(
                        f"{best_name} selected based on the lowest RMSE."
                    )

                    st.subheader("📈 Model Comparison")
                    st.dataframe(
                        results_df.style.format({
                            "MAE": "{:.4f}",
                            "RMSE": "{:.4f}",
                            "R² Score": "{:.4f}"
                        }),
                        use_container_width=True
                    )

                    st.subheader("Actual vs Predicted")
                    chart_df = pd.DataFrame({
                        "Actual": np.asarray(y_test),
                        "Predicted": np.asarray(best_pred)
                    })

                    fig = px.scatter(
                        chart_df,
                        x="Actual",
                        y="Predicted",
                        title=f"Actual vs Predicted — {best_name}",
                        trendline=None
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Prediction Errors")
                    error_df = pd.DataFrame({
                        "Actual": np.asarray(y_test),
                        "Predicted": np.asarray(best_pred),
                        "Error": np.asarray(y_test) - np.asarray(best_pred)
                    })
                    st.dataframe(error_df.head(100), use_container_width=True)

                    st.download_button(
                        "⬇️ Download Model Metrics CSV",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="autregress_model_metrics.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Training error: {e}")
                    st.info(
                        "Check that your target column is numeric, "
                        "features are selected, and the dataset has enough valid rows."
                    )

    if "results_df" in st.session_state:
        st.divider()
        st.subheader("Latest Training Results")
        st.dataframe(
            st.session_state["results_df"],
            use_container_width=True
        )

elif page == "Prediction Center":
    st.title("🔮 Prediction Center")

    if "best_model" not in st.session_state:
        st.info("First upload a dataset and train models in Regression Lab.")
    else:
        st.success(f"Active model: {st.session_state['best_name']}")

        features = st.session_state["features"]
        best_model = st.session_state["best_model"]

        st.write("Enter values for a new prediction:")
        input_values = {}

        for feature in features:
            if df is not None and feature in df.columns:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    median_value = pd.to_numeric(
                        df[feature], errors="coerce"
                    ).median()
                    if pd.isna(median_value):
                        median_value = 0.0

                    input_values[feature] = st.number_input(
                        feature,
                        value=float(median_value)
                    )
                else:
                    options = df[feature].dropna().astype(str).unique().tolist()
                    if options:
                        input_values[feature] = st.selectbox(feature, options)
                    else:
                        input_values[feature] = st.text_input(feature)
            else:
                input_values[feature] = st.text_input(feature)

        if st.button("🔍 Predict", type="primary"):
            try:
                input_df = pd.DataFrame([input_values], columns=features)
                prediction = best_model.predict(input_df)[0]
                st.metric(
                    f"Predicted {st.session_state['target']}",
                    f"{prediction:,.4f}"
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")

st.sidebar.divider()
st.sidebar.caption("AutoRegress AI | Machine Learning Platform")
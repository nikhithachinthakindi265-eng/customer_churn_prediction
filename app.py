import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        html {
            font-size: 24px !important;
        }

        :root {
            --pastel-lavender: #7869bd;
            --pastel-lavender-light: #e7e1fb;
            --pastel-mint: #d9eee3;
            --pastel-cream: #ffffff;
            --pastel-ink: #303447;
            --pastel-muted: #74778a;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--pastel-cream);
            color: var(--pastel-ink);
            color-scheme: light;
            font-size: 24px;
        }

        [data-testid="stAppViewContainer"] * {
            color: var(--pastel-ink) !important;
        }

        [data-testid="stHeader"] * {
            color: var(--pastel-ink) !important;
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricDelta"],
        [data-baseweb="tab"],
        [data-baseweb="radio"] label,
        [data-baseweb="select"] *,
        [data-testid="stFileUploader"] *,
        input,
        textarea {
            color: var(--pastel-ink) !important;
        }

        .stApp [data-testid="stMarkdownContainer"] p {
            font-size: 26px !important;
            line-height: 1.7 !important;
        }

        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] td,
        .stApp [data-testid="stMarkdownContainer"] th {
            font-size: 25px !important;
            line-height: 1.6 !important;
        }

        [data-testid="stAppViewContainer"] code {
            background: #6f61b4 !important;
            color: #ffffff !important;
            border-radius: 5px;
            padding: 0.12rem 0.35rem;
        }

        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stWidgetLabel"],
        .stApp label,
        [data-baseweb="tab"],
        [data-baseweb="radio"] label,
        .stApp input,
        .stApp textarea,
        [data-baseweb="select"] {
            font-size: 24px !important;
            line-height: 1.55 !important;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 247, 230, 0.9);
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-right: 3rem;
            padding-bottom: 4rem;
            padding-left: 3rem;
        }

        .main .block-container [data-testid="stVerticalBlock"] {
            gap: 1.25rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e6e3ec;
        }

        section[data-testid="stSidebar"] {
            width: 360px !important;
            min-width: 360px !important;
        }

        h1, h2, h3 {
            color: var(--pastel-ink);
        }

        h1 {
            font-size: 2.45rem !important;
            line-height: 1.2 !important;
        }

        h2 {
            font-size: 2rem !important;
            line-height: 1.3 !important;
        }

        h3 {
            font-size: 1.5rem !important;
            line-height: 1.35 !important;
        }

        [data-testid="stMetric"] {
            background: rgba(231, 225, 251, 0.82);
            border: 1px solid #d8d0f0;
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 5px 16px rgba(57, 49, 93, 0.07);
        }

        [data-testid="stMetricLabel"] {
            color: var(--pastel-muted);
            font-size: 24px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.1rem !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            background: var(--pastel-lavender);
            color: white !important;
            border: 0;
            border-radius: 10px;
            padding: 0.55rem 1rem;
            font-size: 24px;
            font-weight: 600;
            transition: background 120ms ease, transform 120ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #6f61b4;
            color: white !important;
            border: 0;
            transform: translateY(-1px);
        }

        .stButton > button *,
        .stDownloadButton > button *,
        .stFormSubmitButton > button * {
            color: white !important;
        }

        [data-testid="stSidebar"] .stDownloadButton > button,
        [data-testid="stSidebar"] .stDownloadButton > button * {
            color: #ffffff !important;
            background: #59488f !important;
        }

        [data-testid="stAppViewContainer"] .stDownloadButton button,
        [data-testid="stAppViewContainer"] .stDownloadButton button * {
            color: #ffffff !important;
            background: #59488f !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stButton"] button,
        [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button,
        [data-testid="stAppViewContainer"] [data-testid="stButton"] button *,
        [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button * {
            color: #ffffff !important;
            background: #59488f !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stButton"] button:hover,
        [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button:hover {
            background: #493975 !important;
        }

        [data-baseweb="tab-list"] {
            gap: 8px;
        }

        [data-baseweb="tab"] {
            background: var(--pastel-lavender-light);
            border-radius: 10px 10px 0 0;
            padding: 8px 16px;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            background: var(--pastel-mint);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #dcd5ee;
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stFileUploader"] {
            background: #7869bd;
            border-radius: 14px;
            padding: 12px;
        }

        [data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }

        .stApp [data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label *,
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] *,
        div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
        [data-testid="stFileUploaderDropzone"] * {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .upload-label,
        .stApp [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] label *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            color: #ffffff !important;
            background-color: #59488f !important;
            border-color: #59488f !important;
        }

        [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #6f61b4;
            border: 1px dashed #cfc7f1;
            border-radius: 10px;
        }

        [data-testid="stNumberInput"] {
            background: #7869bd;
            border-radius: 12px;
            padding: 10px 12px;
        }

        [data-testid="stNumberInput"] * {
            color: #ffffff !important;
        }

        [data-testid="stNumberInput"] label,
        [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] *,
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {
            color: #ffffff !important;
        }

        div[data-testid="stElementContainer"]:has([data-testid="stNumberInput"]),
        div[data-testid="stElementContainer"]:has([data-testid="stSelectbox"]) {
            background: #7869bd;
            border-radius: 12px;
            padding: 10px 12px;
        }

        div[data-testid="stElementContainer"]:has([data-testid="stNumberInput"]) [data-testid="stWidgetLabel"] *,
        div[data-testid="stElementContainer"]:has([data-testid="stSelectbox"]) [data-testid="stWidgetLabel"] * {
            color: #ffffff !important;
        }

        [data-testid="stNumberInput"] input {
            color: #ffffff !important;
            background: #111111 !important;
            border: 1px solid #59488f !important;
        }

        .stApp [data-testid="stMarkdownContainer"] .tenure-label {
            color: #ffffff !important;
            background: #7869bd;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 24px;
        }

        .stApp [data-testid="stMarkdownContainer"] .upload-label {
            color: #ffffff !important;
            background: #7869bd;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 24px;
        }

        .stApp [data-testid="stMarkdownContainer"] .prediction-results {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin: 8px 0;
        }

        .stApp [data-testid="stMarkdownContainer"] .prediction-box {
            border-radius: 14px;
            padding: 16px 20px;
        }

        .stApp [data-testid="stMarkdownContainer"] .prediction-results.churn .prediction-box {
            background: #fbe1e1;
            border: 1px solid #e9b2b2;
        }

        .stApp [data-testid="stMarkdownContainer"] .prediction-results.no-churn .prediction-box {
            background: #e2f2e5;
            border: 1px solid #aed3b5;
        }

        .stApp [data-testid="stMarkdownContainer"] .prediction-box span {
            display: block;
            font-size: 22px;
            margin-bottom: 5px;
        }

        .stApp [data-testid="stMarkdownContainer"] .prediction-box strong {
            font-size: 1.5rem;
        }

        @media (max-width: 700px) {
            .stApp [data-testid="stMarkdownContainer"] .prediction-results {
                grid-template-columns: 1fr;
            }
        }

        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border: 1px solid #59488f !important;
            border-color: #59488f !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color: #59488f !important;
            box-shadow: 0 0 0 1px #59488f !important;
        }

        [data-testid="stNumberInput"] input:focus,
        [data-testid="stSelectbox"] [role="combobox"]:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        [data-testid="stSelectbox"] {
            background: #7869bd;
            border-radius: 12px;
            padding: 10px 12px;
        }

        [data-testid="stSelectbox"] *,
        [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #111111 !important;
            border-color: #59488f !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] * {
            color: #ffffff !important;
        }

        [data-baseweb="popover"] [role="listbox"] {
            background: #111111 !important;
        }

        [data-baseweb="popover"] [role="option"] {
            color: #ffffff !important;
        }

        [data-baseweb="popover"] [role="option"] * {
            color: #ffffff !important;
        }

        [data-baseweb="popover"] [role="option"]:hover {
            background: #292929 !important;
        }

        [data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid rgba(120, 105, 189, 0.16);
            padding: 0.9rem 1.1rem;
        }

        @media (max-width: 700px) {
            .main .block-container {
                padding: 1.5rem 1rem 3rem;
            }

            section[data-testid="stSidebar"] {
                width: 85vw !important;
                min-width: 85vw !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Customer Churn Predictor")
st.write("A student project for exploring customer churn and making simple predictions.")

page = st.radio("Go to", ["Home", "Explore data", "Train model", "Make prediction"], horizontal=True)
data_path = Path(__file__).parent / "data" / "ibm_telco_churn.csv"
ibm_customers = pd.read_csv(data_path)
column_mapping = {
    "customerID": "customer_id",
    "tenure": "tenure_months",
    "MonthlyCharges": "monthly_charges",
    "Contract": "contract",
    "InternetService": "internet_service",
    "Churn": "churned",
}
sample_customers = ibm_customers.rename(columns=column_mapping)
sample_customers["churned"] = sample_customers["churned"].map({"No": 0, "Yes": 1})
required_columns = [
    "customer_id", "tenure_months", "monthly_charges", "contract",
    "internet_service", "churned",
]
sample_customers = sample_customers[required_columns].copy()
dataset_is_uploaded = False
customers = sample_customers.copy()

with st.sidebar:
    st.subheader("Customer data")
    st.caption("Using IBM Telco data by default. You can upload a labeled CSV with the same columns.")
    st.download_button(
        "Download sample CSV",
        data=sample_customers.to_csv(index=False),
        file_name="customer_churn_template.csv",
        mime="text/csv",
    )
    st.markdown('<div class="upload-label">Upload your customer CSV</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your customer CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        uploaded_customers = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        missing_columns = [column for column in required_columns if column not in uploaded_customers.columns]
        if missing_columns:
            st.sidebar.error("Missing columns: " + ", ".join(missing_columns))
        else:
            uploaded_customers = uploaded_customers[required_columns].copy()
            number_columns = ["tenure_months", "monthly_charges", "churned"]
            for column in number_columns:
                uploaded_customers[column] = pd.to_numeric(uploaded_customers[column], errors="coerce")

            text_columns = ["customer_id", "contract", "internet_service"]
            for column in text_columns:
                uploaded_customers[column] = uploaded_customers[column].astype("string").str.strip()

            if uploaded_customers.isna().any().any():
                st.sidebar.error("The CSV has blank or non-numeric values in required fields.")
            elif uploaded_customers[text_columns].eq("").any().any():
                st.sidebar.error("Customer ID, contract, and internet service cannot be blank.")
            elif len(uploaded_customers) < 8:
                st.sidebar.error("Add at least 8 customer rows so the app can split data for model training.")
            elif not set(uploaded_customers["churned"].unique()).issubset({0, 1}):
                st.sidebar.error("The churned column must contain only 0 (stayed) and 1 (churned).")
            elif uploaded_customers["churned"].nunique() != 2:
                st.sidebar.error("Include examples of both churned (1) and stayed (0) customers.")
            elif uploaded_customers["churned"].value_counts().min() < 2:
                st.sidebar.error("Include at least two rows for each churned/stayed outcome.")
            else:
                uploaded_customers["contract"] = uploaded_customers["contract"].astype(str)
                uploaded_customers["internet_service"] = uploaded_customers["internet_service"].astype(str)
                customers = uploaded_customers
                dataset_is_uploaded = True
                st.sidebar.success(f"Using your CSV ({len(customers)} customers).")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
        st.sidebar.error("Could not read that file as a CSV. The built-in sample is still in use.")


def build_model(classifier=None):
    numeric_columns = ["tenure_months", "monthly_charges"]
    categorical_columns = ["contract", "internet_service"]
    preprocess = ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numeric_columns),
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier if classifier is not None else LogisticRegression(max_iter=1000)),
        ]
    )

if page == "Home":
    st.info("Welcome! Start by exploring the sample data, then try the beginner model.")
    st.subheader("Project roadmap")
    first, second, third = st.columns(3)
    with first:
        st.markdown("### 1. Explore data")
        st.caption("Understand customer records and patterns.")
    with second:
        st.markdown("### 2. Train a model")
        st.caption("Use scikit-learn to learn churn patterns.")
    with third:
        st.markdown("### 3. Make predictions")
        st.caption("Enter customer details to get a prediction.")
elif page == "Explore data":
    st.header("Explore customer data")
    if dataset_is_uploaded:
        st.caption("Showing the CSV you uploaded.")
    else:
        st.caption("Showing the IBM Telco Customer Churn dataset.")

    churn_count = int(customers["churned"].sum())
    total_customers = len(customers)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Customers", total_customers)
    metric2.metric("Churned", churn_count)
    metric3.metric("Churn rate", f"{churn_count / total_customers:.0%}")

    st.subheader("Customer records")
    st.dataframe(customers, use_container_width=True, hide_index=True)

    st.subheader("Churn by contract type")
    churn_by_contract = customers.groupby("contract", as_index=False)["churned"].mean()
    churn_by_contract = churn_by_contract.rename(columns={"churned": "churn_rate"})
    chart = px.bar(
        churn_by_contract,
        x="contract",
        y="churn_rate",
        text=churn_by_contract["churn_rate"].map("{:.0%}".format),
        labels={"contract": "Contract", "churn_rate": "Churn rate"},
        title="Share of customers who churned",
    )
    chart.update_layout(
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#000000"},
        font_size=23,
        title={"font": {"color": "#000000"}},
        xaxis={"tickfont": {"color": "#000000"}, "title": {"font": {"color": "#000000"}}},
        yaxis={"tickfont": {"color": "#000000"}, "title": {"font": {"color": "#000000"}}},
    )
    chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(chart, use_container_width=True)
elif page == "Train model":
    st.header("Train a beginner model")
    st.caption("Compare Logistic Regression and Random Forest on the same held-out customers.")
    if dataset_is_uploaded:
        st.info(f"Training with your uploaded CSV ({len(customers)} rows). Scores depend on the quality and representativeness of your data.")
    else:
        st.info("Training with the IBM Telco Customer Churn dataset (7,043 customer rows). Scores depend on the data and this single train-test split.")

    features = customers.drop(columns=["customer_id", "churned"])
    target = customers["churned"]

    st.write(
        "The app keeps the target (`churned`) separate from the input features. "
        "It splits the rows into training and test groups, converts contract and "
        "internet service into numeric values, then trains the model."
    )
    with st.expander("What do these steps mean?"):
        st.markdown(
            """
            - **Target:** `churned` is the answer the model is learning to predict (`1` means churn, `0` means stay).
            - **Features:** tenure, monthly charges, contract, and internet service are the information given to the model.
            - **Training rows:** the model uses these rows to learn patterns. The remaining **test rows** are held back to check its predictions.
            - **Preprocessing:** numeric values are scaled, and text categories are converted into a format the model can use.
            """
        )

    comparison_threshold = 0.50
    st.caption("Both models use the same 75/25 stratified split and a fixed 50% classification threshold for this comparison.")

    if st.button("Train and evaluate model", type="primary"):
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=42,
            stratify=target,
        )

        models = {
            "Logistic Regression": build_model(),
            "Random Forest": build_model(
                RandomForestClassifier(n_estimators=300, random_state=42)
            ),
        }
        evaluation = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= comparison_threshold).astype(int)
            evaluation[model_name] = {
                "model": model,
                "probabilities": probabilities,
                "predictions": predictions,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }

        logistic_result = evaluation["Logistic Regression"]
        logistic_model = logistic_result["model"]
        logistic_probabilities = logistic_result["probabilities"]
        logistic_predictions = logistic_result["predictions"]

        st.success(
            f"Both models trained using {len(x_train)} rows and evaluated on {len(x_test)} rows "
            f"with a fixed {comparison_threshold:.0%} classification threshold."
        )
        for model_name, result in evaluation.items():
            st.subheader(f"{model_name} scores")
            score_columns = st.columns(5)
            score_columns[0].metric("Accuracy", f"{result['accuracy']:.0%}")
            score_columns[1].metric("Precision", f"{result['precision']:.0%}")
            score_columns[2].metric("Recall", f"{result['recall']:.0%}")
            score_columns[3].metric("F1-score", f"{result['f1']:.0%}")
            score_columns[4].metric("ROC-AUC", f"{result['roc_auc']:.0%}")
        with st.expander("How should I read these scores?"):
            st.markdown(
            f"""
                - **Accuracy:** the share of test predictions that were correct overall.
                - **Precision:** of the customers predicted to churn, the share who actually churned.
                - **Recall:** of the customers who actually churned, the share the model found.
                - **F1-score:** a single score that balances precision and recall.
                - **ROC-AUC:** how well the model ranks churners above customers who stayed, across different thresholds. A score near 50% is similar to random ranking; 100% is perfect ranking.

                Here, churn (`1`) is the positive result. Both models use the same {len(y_test)} held-back test rows. The threshold stays fixed at 50% so the test set is not used to tune it.
                """
            )

        roc_chart = px.line(title="ROC curves on the same held-out test set")
        for model_name, result in evaluation.items():
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test, result["probabilities"]
            )
            roc_chart.add_scatter(
                x=false_positive_rate,
                y=true_positive_rate,
                mode="lines",
                name=model_name,
            )
        roc_chart.update_layout(
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        roc_chart.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line={"dash": "dash", "color": "#8a8796"},
        )
        roc_chart.update_layout(
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#000000"},
            font_size=23,
            title={"font": {"color": "#000000"}},
            legend={"font": {"color": "#000000"}},
            xaxis={
                "range": [0, 1],
                "title": {"font": {"color": "#000000"}},
                "tickfont": {"color": "#000000"},
            },
            yaxis={
                "range": [0, 1],
                "scaleanchor": "x",
                "scaleratio": 1,
                "title": {"font": {"color": "#000000"}},
                "tickfont": {"color": "#000000"},
            },
        )
        st.plotly_chart(roc_chart, use_container_width=True)

        st.subheader("Confusion matrices")
        matrix_columns = st.columns(2)
        for matrix_column, (model_name, result) in zip(matrix_columns, evaluation.items()):
            matrix_column.markdown(f"**{model_name}**")
            matrix = confusion_matrix(y_test, result["predictions"], labels=[0, 1])
            matrix_table = pd.DataFrame(
                matrix,
                index=["Actual: no churn", "Actual: churn"],
                columns=["Predicted: no churn", "Predicted: churn"],
            )
            matrix_column.dataframe(matrix_table, use_container_width=True)
        st.caption("Each matrix uses the same held-out test customers. Rows are actual classes; columns are predicted classes.")

        with st.expander("Review individual test predictions"):
            review = pd.DataFrame(
                {
                    "customer_id": customers.loc[y_test.index, "customer_id"].to_numpy(),
                    "actual_churned": y_test.to_numpy(),
                    "predicted_churn": logistic_predictions,
                    "churn_probability": logistic_probabilities.round(3),
                    "correct": (y_test.to_numpy() == logistic_predictions),
                }
            )
            st.caption("The table shows only the held-back test rows. Churn is 1; staying is 0.")
            st.dataframe(review, use_container_width=True, hide_index=True)

        st.subheader("What patterns did Logistic Regression learn?")
        st.caption(
            "A positive coefficient moves the estimate toward churn; a negative one moves it toward staying. "
            "These are patterns in this dataset, not proof that a feature causes churn."
        )
        preprocessor = logistic_model.named_steps["preprocess"]
        feature_names = [
            name.replace("numbers__", "").replace("categories__", "")
            for name in preprocessor.get_feature_names_out()
        ]
        coefficients = logistic_model.named_steps["classifier"].coef_[0]
        coefficient_data = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "direction": [
                    "Higher churn estimate" if value >= 0 else "Lower churn estimate"
                    for value in coefficients
                ],
            }
        ).sort_values("coefficient")
        coefficient_chart = px.bar(
            coefficient_data,
            x="coefficient",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Higher churn estimate": "#c66f83",
                "Lower churn estimate": "#397258",
            },
            labels={"coefficient": "Model coefficient", "feature": "Input", "direction": "Direction"},
        )
        coefficient_chart.update_layout(
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#171717"},
            font_size=23,
            height=360,
            legend_title_text="Association",
            legend={"font": {"color": "#171717"}, "title": {"font": {"color": "#171717"}}},
            xaxis={"tickfont": {"color": "#171717"}, "title": {"font": {"color": "#171717"}}},
            yaxis={"tickfont": {"color": "#171717"}, "title": {"font": {"color": "#171717"}}},
        )
        coefficient_chart.add_vline(x=0, line_color="#74778a", line_width=1)
        st.plotly_chart(coefficient_chart, use_container_width=True)
else:
    st.header("Make a prediction")
    st.warning("This prediction is for learning. Check that your data is representative before using results for any real decisions.")
    threshold = st.slider("Churn alert threshold", 0.05, 0.95, 0.50, step=0.05)
    st.caption("Lowering the threshold marks more customers as higher risk; raising it marks fewer.")

    single_tab, batch_tab = st.tabs(["Single customer", "Batch CSV"])
    contract_options = sorted(customers["contract"].astype(str).unique())
    internet_options = sorted(customers["internet_service"].astype(str).unique())

    with single_tab:
        with st.form("prediction_form"):
            st.markdown('<div class="tenure-label">Tenure (months)</div>', unsafe_allow_html=True)
            tenure_months = st.number_input(
                "Tenure (months)", min_value=0, max_value=120, value=12, label_visibility="collapsed"
            )
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=70.0)
            contract = st.selectbox("Contract type", contract_options)
            internet_service = st.selectbox("Internet service", internet_options)
            submitted = st.form_submit_button("Estimate churn")

        if submitted:
            features = customers.drop(columns=["customer_id", "churned"])
            target = customers["churned"]
            model = build_model()
            model.fit(features, target)

            new_customer = pd.DataFrame(
                [{
                    "tenure_months": tenure_months,
                    "monthly_charges": monthly_charges,
                    "contract": contract,
                    "internet_service": internet_service,
                }]
            )
            churn_probability = model.predict_proba(new_customer)[0][1]
            prediction_label = "Churn" if churn_probability >= threshold else "No Churn"
            result_class = "churn" if prediction_label == "Churn" else "no-churn"
            st.markdown(
                f'<div class="prediction-results {result_class}">'
                f'<div class="prediction-box"><span>Prediction</span><strong>{prediction_label}</strong></div>'
                f'<div class="prediction-box"><span>Churn probability</span><strong>{churn_probability:.1%}</strong></div>'
                "</div>",
                unsafe_allow_html=True,
            )

    with batch_tab:
        st.write("Upload customer details without a `churned` column. The model is trained from the labeled dataset currently selected in the sidebar.")
        prediction_columns = [
            "customer_id", "tenure_months", "monthly_charges", "contract",
            "internet_service",
        ]
        blank_template = pd.DataFrame(columns=prediction_columns)
        st.download_button(
            "Download prediction CSV template",
            data=blank_template.to_csv(index=False),
            file_name="customers_to_score.csv",
            mime="text/csv",
        )
        st.markdown('<div class="upload-label">Upload customers to score</div>', unsafe_allow_html=True)
        batch_file = st.file_uploader(
            "Upload customers to score", type=["csv"], key="batch_prediction_csv", label_visibility="collapsed"
        )

        if batch_file is not None:
            try:
                batch_customers = pd.read_csv(BytesIO(batch_file.getvalue()))
                missing_columns = [column for column in prediction_columns if column not in batch_customers.columns]
                if missing_columns:
                    st.error("Missing columns: " + ", ".join(missing_columns))
                elif batch_customers.empty:
                    st.error("Add at least one customer row to the CSV.")
                else:
                    batch_customers = batch_customers[prediction_columns].copy()
                    numeric_columns = ["tenure_months", "monthly_charges"]
                    text_columns = ["customer_id", "contract", "internet_service"]
                    for column in numeric_columns:
                        batch_customers[column] = pd.to_numeric(batch_customers[column], errors="coerce")
                    for column in text_columns:
                        batch_customers[column] = batch_customers[column].astype("string").str.strip()

                    if batch_customers.isna().any().any() or batch_customers[text_columns].eq("").any().any():
                        st.error("Fill in every required field, using numbers for tenure and monthly charges.")
                    else:
                        training_features = customers.drop(columns=["customer_id", "churned"])
                        training_target = customers["churned"]
                        model = build_model()
                        model.fit(training_features, training_target)
                        prediction_features = batch_customers.drop(columns=["customer_id"])
                        probabilities = model.predict_proba(prediction_features)[:, 1]

                        results = batch_customers.copy()
                        results["prediction"] = [
                            "Churn" if probability >= threshold else "No Churn"
                            for probability in probabilities
                        ]
                        results["churn_probability"] = [f"{probability:.1%}" for probability in probabilities]
                        styled_results = results.style.map(
                            lambda value: (
                                "background-color: #fbe1e1; color: #303447"
                                if value == "Churn"
                                else "background-color: #e2f2e5; color: #303447"
                            ),
                            subset=["prediction"],
                        )
                        st.dataframe(styled_results, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download predictions",
                            data=results.to_csv(index=False),
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
                st.error("Could not read that file as a CSV. Please use the template and try again.")


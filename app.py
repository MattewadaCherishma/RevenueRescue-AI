import os
import json
import random
from datetime import datetime

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from dotenv import load_dotenv

# Optional Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Optional Razorpay
try:
    import razorpay
    RAZORPAY_AVAILABLE = True
except Exception:
    RAZORPAY_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RevenueRescue AI",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "transactions.csv"
)

BATCH_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "batch_results.csv"
)

AUDIT_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "audit_trail.csv"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET SECRETS
# ============================================================

def get_secret(name):

    # Streamlit Cloud secrets
    try:
        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass

    # Local .env
    return os.getenv(name)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

RAZORPAY_KEY_ID = get_secret("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = get_secret("RAZORPAY_KEY_SECRET")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_AVAILABLE and GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    except Exception:

        gemini_client = None


# ============================================================
# RAZORPAY CLIENT
# ============================================================

razorpay_client = None

if (
    RAZORPAY_AVAILABLE
    and RAZORPAY_KEY_ID
    and RAZORPAY_KEY_SECRET
):

    try:

        razorpay_client = razorpay.Client(
            auth=(
                RAZORPAY_KEY_ID,
                RAZORPAY_KEY_SECRET
            )
        )

    except Exception:

        razorpay_client = None


# ============================================================
# LOAD TRANSACTION DATA
# ============================================================

@st.cache_data
def load_transactions():

    if not os.path.exists(DATA_PATH):

        st.error(
            f"Dataset not found: {DATA_PATH}"
        )

        st.stop()

    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False
    )

    return df


transactions_df = load_transactions()


# ============================================================
# CLEAN DATA
# ============================================================

def clean_transactions(df):

    df = df.copy()

    # Convert numeric columns
    numeric_columns = [
        "amount",
        "customer_success_rate",
        "previous_failed_attempts",
        "is_new_device",
        "recovered"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove rows with missing critical values
    df = df.dropna(
        subset=[
            "amount",
            "customer_success_rate",
            "previous_failed_attempts",
            "is_new_device",
            "recovered"
        ]
    )

    # Make sure failure_reason is a string
    if "failure_reason" in df.columns:

        df["failure_reason"] = (
            df["failure_reason"]
            .fillna("")
            .astype(str)
        )

    if "payment_method" in df.columns:

        df["payment_method"] = (
            df["payment_method"]
            .fillna("Unknown")
            .astype(str)
        )

    return df


transactions_df = clean_transactions(
    transactions_df
)


# ============================================================
# FAILED TRANSACTIONS
# ============================================================

failed_transactions = transactions_df[
    transactions_df["status"].astype(str).str.lower() == "failed"
].copy()


# ============================================================
# MODEL TRAINING
# ============================================================

FEATURES = [
    "amount",
    "payment_method",
    "customer_success_rate",
    "previous_failed_attempts",
    "is_new_device",
    "failure_reason"
]


@st.cache_resource
def train_recovery_model(df):

    X = df[FEATURES].copy()

    y = df["recovered"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    categorical_features = [
        "payment_method",
        "failure_reason"
    ]

    numerical_features = [
        "amount",
        "customer_success_rate",
        "previous_failed_attempts",
        "is_new_device"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        )
    }

    return pipeline, metrics


with st.spinner("Training Revenue Recovery AI model..."):

    recovery_model, model_metrics = train_recovery_model(
        transactions_df
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_recovery_probability(transaction):

    input_df = pd.DataFrame(
        [{
            column: transaction[column]
            for column in FEATURES
        }]
    )

    probability = recovery_model.predict_proba(
        input_df
    )[0][1]

    return float(probability)


# ============================================================
# AI DECISION ENGINE
# ============================================================

def deterministic_decision(
    transaction,
    probability
):

    amount = float(
        transaction["amount"]
    )

    failure_reason = str(
        transaction["failure_reason"]
    )

    previous_failures = int(
        transaction["previous_failed_attempts"]
    )

    success_rate = float(
        transaction["customer_success_rate"]
    )

    # Safety rules
    if previous_failures >= 4:

        return {
            "action": "DO_NOT_RETRY",
            "reason": (
                "Customer has multiple previous "
                "failed payment attempts."
            )
        }

    if amount > 100000 and probability < 0.70:

        return {
            "action": "ADDITIONAL_AUTHENTICATION",
            "reason": (
                "High-value transaction requires "
                "additional verification."
            )
        }

    if failure_reason == "Expired Card":

        return {
            "action": "UPDATE_PAYMENT_METHOD",
            "reason": (
                "Card appears to be expired. "
                "Customer should update payment method."
            )
        }

    if failure_reason == "Insufficient Funds":

        if probability >= 0.60:

            return {
                "action": "RETRY_LATER",
                "reason": (
                    "Retry later because the customer "
                    "may have insufficient funds currently."
                )
            }

        return {
            "action": "DO_NOT_RETRY",
            "reason": (
                "Low predicted recovery probability "
                "for insufficient-funds failure."
            )
        }

    if failure_reason == "Network Error":

        return {
            "action": "RETRY_NOW",
            "reason": (
                "Network-related failures have a "
                "relatively high retry potential."
            )
        }

    if failure_reason == "Bank Timeout":

        return {
            "action": "RETRY_NOW",
            "reason": (
                "Bank timeout can often be recovered "
                "through a controlled retry."
            )
        }

    if failure_reason == "Authentication Failed":

        if probability >= 0.60:

            return {
                "action": "ADDITIONAL_AUTHENTICATION",
                "reason": (
                    "Additional authentication may "
                    "resolve the payment failure."
                )
            }

        return {
            "action": "DO_NOT_RETRY",
            "reason": (
                "Recovery probability is below "
                "the safe retry threshold."
            )
        }

    if probability >= 0.75:

        return {
            "action": "RETRY_NOW",
            "reason": (
                "High predicted recovery probability."
            )
        }

    if probability >= 0.60:

        return {
            "action": "RETRY_LATER",
            "reason": (
                "Moderate recovery probability; "
                "retry later to avoid aggressive retries."
            )
        }

    if probability >= 0.45:

        return {
            "action": "UPDATE_PAYMENT_METHOD",
            "reason": (
                "Moderate recovery potential; "
                "payment method intervention is safer."
            )
        }

    return {
        "action": "DO_NOT_RETRY",
        "reason": (
            "Recovery probability is too low "
            "for another automated attempt."
        )
    }


def ask_gemini(
    transaction,
    probability
):

    if gemini_client is None:

        return None

    prompt = f"""
You are the decision engine for RevenueRescue AI.

This is a SIMULATION-ONLY payment recovery system.
Do not claim that money was actually recovered.

Analyze this failed payment:

Transaction ID: {transaction["transaction_id"]}
Amount: {transaction["amount"]}
Payment Method: {transaction["payment_method"]}
Customer Success Rate: {transaction["customer_success_rate"]}
Previous Failed Attempts: {transaction["previous_failed_attempts"]}
New Device: {transaction["is_new_device"]}
Failure Reason: {transaction["failure_reason"]}
ML Recovery Probability: {probability:.3f}

Choose exactly ONE action:

RETRY_NOW
RETRY_LATER
UPDATE_PAYMENT_METHOD
ADDITIONAL_AUTHENTICATION
DO_NOT_RETRY

Rules:

1. Never recommend unlimited retries.
2. Do not recommend retrying repeatedly.
3. High previous failure count should reduce retrying.
4. Expired cards should normally use UPDATE_PAYMENT_METHOD.
5. High-value transactions may require ADDITIONAL_AUTHENTICATION.
6. Low recovery probability should normally use DO_NOT_RETRY.
7. The system must remain bounded and safe.

Return ONLY valid JSON:

{{
    "action": "ACTION_NAME",
    "reason": "short explanation"
}}
"""

    try:

        response = gemini_client.models.generate_content(

            model="gemini-3.7-flash",

            contents=prompt
        )

        text = response.text.strip()

        # Remove markdown fences if Gemini adds them
        text = text.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        result = json.loads(text)

        allowed_actions = [
            "RETRY_NOW",
            "RETRY_LATER",
            "UPDATE_PAYMENT_METHOD",
            "ADDITIONAL_AUTHENTICATION",
            "DO_NOT_RETRY"
        ]

        if result.get("action") not in allowed_actions:

            return None

        return result

    except Exception:

        return None


# ============================================================
# SAFETY GUARDRAILS
# ============================================================

def apply_safety_guardrails(
    transaction,
    ai_decision,
    probability
):

    action = ai_decision["action"]

    previous_failures = int(
        transaction["previous_failed_attempts"]
    )

    amount = float(
        transaction["amount"]
    )

    failure_reason = str(
        transaction["failure_reason"]
    )

    # Stop repeated retrying
    if previous_failures >= 4:

        action = "DO_NOT_RETRY"

        reason = (
            "Safety rule: maximum retry exposure "
            "reached for this customer."
        )

        return {
            "action": action,
            "reason": reason,
            "guardrail_triggered": True
        }

    # Expired cards should not be retried
    if failure_reason == "Expired Card":

        action = "UPDATE_PAYMENT_METHOD"

        reason = (
            "Safety rule: expired cards require "
            "payment method update."
        )

        return {
            "action": action,
            "reason": reason,
            "guardrail_triggered": True
        }

    # High value + lower confidence
    if amount > 100000 and probability < 0.70:

        action = "ADDITIONAL_AUTHENTICATION"

        reason = (
            "Safety rule: high-value transaction "
            "requires additional authentication."
        )

        return {
            "action": action,
            "reason": reason,
            "guardrail_triggered": True
        }

    return {
        "action": action,
        "reason": ai_decision["reason"],
        "guardrail_triggered": False
    }


# ============================================================
# AUDIT TRAIL
# ============================================================

AUDIT_COLUMNS = [
    "timestamp",
    "transaction_id",
    "customer_id",
    "amount",
    "failure_reason",
    "recovery_probability",
    "ai_action",
    "final_action",
    "guardrail_triggered",
    "execution_status",
    "razorpay_test_order_id",
    "simulated_recovered",
    "notes"
]


def load_audit_trail():

    if not os.path.exists(AUDIT_PATH):

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )

    try:

        df = pd.read_csv(
            AUDIT_PATH,
            keep_default_na=False
        )

        for column in AUDIT_COLUMNS:

            if column not in df.columns:

                df[column] = ""

        return df[AUDIT_COLUMNS]

    except Exception:

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )


def write_audit_record(record):

    os.makedirs(
        os.path.dirname(AUDIT_PATH),
        exist_ok=True
    )

    audit_df = load_audit_trail()

    new_record = {
        column: record.get(
            column,
            ""
        )
        for column in AUDIT_COLUMNS
    }

    audit_df = pd.concat(
        [
            audit_df,
            pd.DataFrame([new_record])
        ],
        ignore_index=True
    )

    audit_df.to_csv(
        AUDIT_PATH,
        index=False
    )


# ============================================================
# RAZORPAY TEST ORDER
# ============================================================

def create_razorpay_test_order(
    amount
):

    if razorpay_client is None:

        return None, (
            "Razorpay Test Mode credentials "
            "are not configured."
        )

    try:

        # Razorpay amount is in paise
        amount_paise = int(
            round(
                float(amount) * 100
            )
        )

        order = razorpay_client.order.create(
            {
                "amount": amount_paise,
                "currency": "INR",
                "receipt": (
                    "rr_" +
                    datetime.now().strftime(
                        "%Y%m%d%H%M%S"
                    )
                )
            }
        )

        return order["id"], "SUCCESS"

    except Exception as e:

        return None, str(e)


# ============================================================
# SIMULATED RECOVERY EXECUTION
# ============================================================

def simulate_recovery(action):

    # These probabilities are ONLY simulation assumptions.
    # They do not represent real payment recovery rates.

    success_probability = {

        "RETRY_NOW": 0.70,

        "RETRY_LATER": 0.55,

        "UPDATE_PAYMENT_METHOD": 0.65,

        "ADDITIONAL_AUTHENTICATION": 0.60,

        "DO_NOT_RETRY": 0.00
    }

    probability = success_probability.get(
        action,
        0.00
    )

    recovered = random.random() < probability

    return recovered


# ============================================================
# BATCH RESULTS
# ============================================================

def load_batch_results():

    if not os.path.exists(BATCH_PATH):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            BATCH_PATH,
            keep_default_na=False
        )

    except Exception:

        return pd.DataFrame()


batch_results = load_batch_results()


# ============================================================
# HEADER
# ============================================================

st.title(
    "💰 RevenueRescue AI"
)

st.subheader(
    "AI-Powered Payment Recovery System"
)

st.info(
    """
    ⚠️ **SIMULATION-ONLY DEMO**

    This application demonstrates an AI-assisted payment
    recovery workflow using historical transaction data,
    machine-learning predictions, bounded decision rules,
    and Razorpay Test Mode.

    No real customer payments or real money are processed.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "RevenueRescue AI"
)

st.sidebar.markdown(
    """
### System Pipeline

1. Payment failure detection
2. Recovery probability prediction
3. AI decision
4. Safety guardrails
5. Recovery action
6. Razorpay Test Mode
7. Audit trail
8. Recovery analytics
"""
)

st.sidebar.success(
    "Model loaded successfully"
)

if gemini_client:

    st.sidebar.success(
        "Gemini AI: Connected"
    )

else:

    st.sidebar.warning(
        "Gemini AI: Fallback mode"
    )

if razorpay_client:

    st.sidebar.success(
        "Razorpay: Test Mode"
    )

else:

    st.sidebar.warning(
        "Razorpay: Not configured"
    )


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_transactions = len(
    transactions_df
)

failed_count = len(
    failed_transactions
)

successful_count = (
    total_transactions -
    failed_count
)

total_transaction_value = (
    transactions_df["amount"]
    .sum()
)

revenue_at_risk = (
    failed_transactions["amount"]
    .sum()
)

historical_recovered_revenue = (
    failed_transactions[
        failed_transactions["recovered"] == 1
    ]["amount"]
    .sum()
)

historical_recovery_rate = (

    (
        failed_transactions["recovered"]
        .sum()
        /
        failed_count
    )
    * 100

    if failed_count > 0

    else 0
)


st.header(
    "📊 Revenue Recovery Overview"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Failed Payments",
    f"{failed_count:,}"
)

col3.metric(
    "Revenue at Risk",
    f"₹{revenue_at_risk:,.2f}"
)

col4.metric(
    "Historical Recovery Rate",
    f"{historical_recovery_rate:.2f}%"
)


st.divider()


# ============================================================
# SECOND METRICS ROW
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Successful Payments",
    f"{successful_count:,}"
)

col2.metric(
    "Total Transaction Value",
    f"₹{total_transaction_value:,.2f}"
)

col3.metric(
    "Historically Recovered Revenue",
    f"₹{historical_recovered_revenue:,.2f}"
)

col4.metric(
    "Unrecovered Revenue",
    f"₹{revenue_at_risk - historical_recovered_revenue:,.2f}"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🤖 Recovery Agent",
        "📈 Analytics",
        "🧾 Audit Trail"
    ]
)


# ============================================================
# TAB 1 - RECOVERY AGENT
# ============================================================

with tab1:

    st.header(
        "🤖 AI Recovery Agent"
    )

    if failed_transactions.empty:

        st.warning(
            "No failed transactions found."
        )

    else:

        transaction_ids = (
            failed_transactions[
                "transaction_id"
            ]
            .astype(str)
            .tolist()
        )

        selected_transaction_id = st.selectbox(
            "Select a failed payment",
            transaction_ids
        )

        selected_transaction = (
            failed_transactions[
                failed_transactions[
                    "transaction_id"
                ].astype(str)
                ==
                selected_transaction_id
            ]
            .iloc[0]
        )

        st.subheader(
            "Payment Information"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Transaction Amount",
            f"₹{float(selected_transaction['amount']):,.2f}"
        )

        c2.metric(
            "Failure Reason",
            str(
                selected_transaction[
                    "failure_reason"
                ]
            )
        )

        c3.metric(
            "Previous Failures",
            int(
                selected_transaction[
                    "previous_failed_attempts"
                ]
            )
        )

        st.write(
            "**Transaction ID:**",
            selected_transaction[
                "transaction_id"
            ]
        )

        st.write(
            "**Customer ID:**",
            selected_transaction[
                "customer_id"
            ]
        )

        st.write(
            "**Payment Method:**",
            selected_transaction[
                "payment_method"
            ]
        )

        st.write(
            "**Customer Success Rate:**",
            f"{float(selected_transaction['customer_success_rate']) * 100:.2f}%"
        )

        st.divider()

        if st.button(
            "🔍 Analyze Payment with AI",
            type="primary"
        ):

            recovery_probability = (
                predict_recovery_probability(
                    selected_transaction
                )
            )

            st.session_state[
                "recovery_probability"
            ] = recovery_probability

            # Gemini decision
            gemini_decision = ask_gemini(
                selected_transaction,
                recovery_probability
            )

            if gemini_decision:

                ai_decision = gemini_decision

                decision_source = (
                    "Gemini AI"
                )

            else:

                ai_decision = deterministic_decision(
                    selected_transaction,
                    recovery_probability
                )

                decision_source = (
                    "Deterministic AI Fallback"
                )

            final_decision = apply_safety_guardrails(
                selected_transaction,
                ai_decision,
                recovery_probability
            )

            st.session_state[
                "ai_decision"
            ] = ai_decision

            st.session_state[
                "final_decision"
            ] = final_decision

            st.session_state[
                "selected_transaction"
            ] = selected_transaction

            st.session_state[
                "decision_source"
            ] = decision_source

        if "recovery_probability" in st.session_state:

            probability = (
                st.session_state[
                    "recovery_probability"
                ]
            )

            ai_decision = (
                st.session_state[
                    "ai_decision"
                ]
            )

            final_decision = (
                st.session_state[
                    "final_decision"
                ]
            )

            st.subheader(
                "🧠 Recovery Intelligence"
            )

            st.progress(
                probability
            )

            st.metric(
                "Predicted Recovery Probability",
                f"{probability * 100:.2f}%"
            )

            st.write(
                "**Decision Source:**",
                st.session_state[
                    "decision_source"
                ]
            )

            st.subheader(
                "AI Recommendation"
            )

            st.info(
                f"""
                **AI Action:** `{ai_decision["action"]}`

                **Reason:** {ai_decision["reason"]}
                """
            )

            st.subheader(
                "🛡️ Safety Layer"
            )

            if final_decision[
                "guardrail_triggered"
            ]:

                st.warning(
                    f"""
                    Safety guardrail triggered.

                    Final action:
                    **{final_decision["action"]}**

                    Reason:
                    {final_decision["reason"]}
                    """
                )

            else:

                st.success(
                    f"""
                    No additional safety override required.

                    Final action:
                    **{final_decision["action"]}**
                    """
                )

            st.divider()

            st.subheader(
                "⚡ Recovery Action Center"
            )

            final_action = final_decision[
                "action"
            ]

            if final_action == "DO_NOT_RETRY":

                st.warning(
                    "⛔ Recovery stopped. No automated retry will be attempted."
                )

                if st.button(
                    "📝 Record Stop Decision"
                ):

                    write_audit_record(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "transaction_id": selected_transaction["transaction_id"],
                            "customer_id": selected_transaction["customer_id"],
                            "amount": selected_transaction["amount"],
                            "failure_reason": selected_transaction["failure_reason"],
                            "recovery_probability": probability,
                            "ai_action": ai_decision["action"],
                            "final_action": final_action,
                            "guardrail_triggered": final_decision["guardrail_triggered"],
                            "execution_status": "STOPPED",
                            "razorpay_test_order_id": "",
                            "simulated_recovered": False,
                            "notes": final_decision["reason"]
                        }
                    )

                    st.success(
                        "Decision recorded in audit trail."
                    )

            else:

                st.write(
                    f"""
                    Recommended action:

                    **{final_action}**
                    """
                )

                if st.button(
                    "🚀 Execute Recovery Action",
                    type="primary"
                ):

                    razorpay_order_id = ""

                    execution_status = (
                        "SIMULATION_EXECUTED"
                    )

                    # ------------------------------------------------
                    # Razorpay Test Mode
                    # ------------------------------------------------

                    if razorpay_client:

                        order_id, result = (
                            create_razorpay_test_order(
                                selected_transaction[
                                    "amount"
                                ]
                            )
                        )

                        if order_id:

                            razorpay_order_id = (
                                order_id
                            )

                            execution_status = (
                                "TEST_ORDER_CREATED"
                            )

                            st.success(
                                "✅ Razorpay Test Mode order created."
                            )

                            st.code(
                                razorpay_order_id
                            )

                        else:

                            st.warning(
                                "Razorpay Test Mode order could not be created."
                            )

                            st.caption(
                                str(result)
                            )

                    # ------------------------------------------------
                    # Simulation
                    # ------------------------------------------------

                    simulated_recovered = (
                        simulate_recovery(
                            final_action
                        )
                    )

                    write_audit_record(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "transaction_id": selected_transaction["transaction_id"],
                            "customer_id": selected_transaction["customer_id"],
                            "amount": selected_transaction["amount"],
                            "failure_reason": selected_transaction["failure_reason"],
                            "recovery_probability": probability,
                            "ai_action": ai_decision["action"],
                            "final_action": final_action,
                            "guardrail_triggered": final_decision["guardrail_triggered"],
                            "execution_status": execution_status,
                            "razorpay_test_order_id": razorpay_order_id,
                            "simulated_recovered": simulated_recovered,
                            "notes": "Simulation-only recovery execution."
                        }
                    )

                    if simulated_recovered:

                        st.success(
                            f"""
                            💰 **Recovery action simulated successfully.**

                            Simulated rescued amount:
                            **₹{float(selected_transaction["amount"]):,.2f}**

                            This is NOT real recovered money.
                            """
                        )

                    else:

                        st.error(
                            """
                            Recovery action was simulated but
                            the payment was not recovered in the simulation.
                            """
                        )

                    st.info(
                        """
                        🔒 Audit record created.

                        Any Razorpay Order ID shown above is from
                        **Razorpay Test Mode only**.
                        """
                    )


# ============================================================
# TAB 2 - ANALYTICS
# ============================================================

with tab2:

    st.header(
        "📈 Revenue Recovery Analytics"
    )

    st.subheader(
        "Failure Reason Analysis"
    )

    reason_analysis = (
        failed_transactions
        .groupby("failure_reason")
        .agg(
            failed_payments=(
                "transaction_id",
                "count"
            ),
            revenue_at_risk=(
                "amount",
                "sum"
            ),
            historically_recovered=(
                "recovered",
                "sum"
            )
        )
        .reset_index()
    )

    reason_analysis[
        "historical_recovery_rate"
    ] = (

        reason_analysis[
            "historically_recovered"
        ]
        /
        reason_analysis[
            "failed_payments"
        ]
        * 100
    )

    reason_analysis = (
        reason_analysis
        .sort_values(
            "revenue_at_risk",
            ascending=False
        )
    )

    st.dataframe(
        reason_analysis,
        use_container_width=True
    )

    st.subheader(
        "Failure Reason Revenue at Risk"
    )

    chart_data = (
        reason_analysis[
            [
                "failure_reason",
                "revenue_at_risk"
            ]
        ]
        .set_index(
            "failure_reason"
        )
    )

    st.bar_chart(
        chart_data
    )

    st.subheader(
        "Historical Recovery Performance"
    )

    recovered_count = int(
        failed_transactions[
            "recovered"
        ].sum()
    )

    unrecovered_count = (
        failed_count -
        recovered_count
    )

    performance_df = pd.DataFrame(
        {
            "Status": [
                "Recovered",
                "Unrecovered"
            ],
            "Payments": [
                recovered_count,
                unrecovered_count
            ]
        }
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "🤖 Model Evaluation"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{model_metrics['accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{model_metrics['precision'] * 100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{model_metrics['recall'] * 100:.2f}%"
    )

    c4.metric(
        "F1 Score",
        f"{model_metrics['f1'] * 100:.2f}%"
    )

    c5.metric(
        "ROC-AUC",
        f"{model_metrics['roc_auc']:.3f}"
    )

    st.caption(
        "Metrics are calculated on a held-out test set."
    )

    cm = model_metrics[
        "confusion_matrix"
    ]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1"
        ]
    )

    st.write(
        "**Confusion Matrix**"
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


# ============================================================
# TAB 3 - AUDIT TRAIL
# ============================================================

with tab3:

    st.header(
        "🧾 Recovery Audit Trail"
    )

    audit_df = load_audit_trail()

    if audit_df.empty:

        st.info(
            """
            No recovery actions have been executed yet.

            Analyze a failed payment and execute a recovery
            action to create an audit record.
            """
        )

    else:

        st.metric(
            "Total Audit Records",
            len(audit_df)
        )

        st.dataframe(
            audit_df.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            height=500
        )

        st.download_button(
            label="⬇️ Download Audit Trail",
            data=audit_df.to_csv(
                index=False
            ),
            file_name="audit_trail.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
RevenueRescue AI | AI-Powered Payment Recovery System

Built for the Razorpay AI Buildathon.

⚠️ Demonstration and simulation only.
No real payments or real customer money are processed.
"""
)

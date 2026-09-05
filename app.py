import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from google import genai

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

import razorpay


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RevenueRescue AI",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

gemini_api_key = None

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    gemini_api_key = os.getenv("GEMINI_API_KEY")


gemini_client = None

if gemini_api_key:

    try:

        gemini_client = genai.Client(
            api_key=gemini_api_key
        )

    except Exception:

        gemini_client = None


# ============================================================
# RAZORPAY TEST MODE CONFIGURATION
# ============================================================

razorpay_key_id = None
razorpay_key_secret = None


try:

    razorpay_key_id = st.secrets["RAZORPAY_KEY_ID"]
    razorpay_key_secret = st.secrets["RAZORPAY_KEY_SECRET"]

except Exception:

    razorpay_key_id = os.getenv(
        "RAZORPAY_KEY_ID"
    )

    razorpay_key_secret = os.getenv(
        "RAZORPAY_KEY_SECRET"
    )


razorpay_client = None


if razorpay_key_id and razorpay_key_secret:

    try:

        razorpay_client = razorpay.Client(
            auth=(
                razorpay_key_id,
                razorpay_key_secret
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
            f"Transaction dataset not found:\n{DATA_PATH}"
        )

        return pd.DataFrame()

    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False
    )

    return df


transactions_df = load_transactions()


# ============================================================
# TRAIN ML MODEL
# ============================================================

@st.cache_resource
def train_recovery_model(df):

    features = [

        "amount",

        "payment_method",

        "customer_success_rate",

        "previous_failed_attempts",

        "is_new_device",

        "failure_reason"

    ]

    X = df[features]

    y = df["recovered"].astype(int)


    categorical_features = [

        "payment_method",

        "failure_reason"

    ]


    numeric_features = [

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
                "numeric",

                "passthrough",

                numeric_features
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


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

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

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            )

    }


    return pipeline, metrics


if not transactions_df.empty:

    recovery_model, model_metrics = train_recovery_model(
        transactions_df
    )

else:

    recovery_model = None

    model_metrics = {}


# ============================================================
# LOAD BATCH RESULTS
# ============================================================

@st.cache_data
def load_batch_results():

    if os.path.exists(BATCH_PATH):

        return pd.read_csv(
            BATCH_PATH
        )

    return pd.DataFrame()


batch_df = load_batch_results()


# ============================================================
# LOAD AUDIT TRAIL
# ============================================================

def load_audit_trail():

    if not os.path.exists(AUDIT_PATH):

        return pd.DataFrame()


    audit_df = pd.read_csv(
        AUDIT_PATH,
        keep_default_na=False
    )


    # --------------------------------------------------------
    # FIX OLD AUDIT FILE
    # --------------------------------------------------------

    if "razorpay_test_order_id" not in audit_df.columns:

        audit_df["razorpay_test_order_id"] = ""

        audit_df.to_csv(
            AUDIT_PATH,
            index=False
        )


    return audit_df


# ============================================================
# RAZORPAY TEST ORDER CREATION
# ============================================================

def create_razorpay_test_order(transaction):

    """
    Creates a Razorpay TEST MODE order.

    This does NOT process real money.
    """

    if razorpay_client is None:

        return {

            "success": False,

            "order_id": None,

            "message":
                "Razorpay Test API credentials are not configured."

        }


    try:

        amount_in_paise = int(
            round(
                float(
                    transaction["amount"]
                ) * 100
            )
        )


        order_data = {

            "amount":
                amount_in_paise,

            "currency":
                "INR",

            "receipt":
                str(
                    transaction[
                        "transaction_id"
                    ]
                ),

            "notes": {

                "source":
                    "RevenueRescue AI",

                "transaction_id":
                    str(
                        transaction[
                            "transaction_id"
                        ]
                    ),

                "recovery_workflow":
                    "AI_PAYMENT_RECOVERY",

                "environment":
                    "TEST_MODE"

            }

        }


        order = razorpay_client.order.create(
            data=order_data
        )


        return {

            "success": True,

            "order_id":
                order.get("id"),

            "message":
                "Razorpay Test Mode order created successfully."

        }


    except Exception as e:

        return {

            "success": False,

            "order_id": None,

            "message":
                f"Razorpay Test API error: {str(e)}"

        }


# ============================================================
# WRITE AUDIT RECORD
# ============================================================

def write_audit_record(

    transaction,

    ml_probability,

    ai_result,

    guardrail_result,

    action_executed,

    recovery_result,

    revenue_rescued,

    razorpay_order_id=""

):


    audit_record = {

        "timestamp":

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),


        "transaction_id":

            transaction[
                "transaction_id"
            ],


        "amount":

            float(
                transaction["amount"]
            ),


        "failure_reason":

            transaction[
                "failure_reason"
            ],


        "previous_failed_attempts":

            int(
                transaction[
                    "previous_failed_attempts"
                ]
            ),


        "ml_recovery_probability":

            float(
                ml_probability
            ),


        "ai_recommendation":

            ai_result.get(
                "action",
                ""
            ),


        "ai_priority":

            ai_result.get(
                "priority",
                ""
            ),


        "ai_fallback_used":

            ai_result.get(
                "fallback_used",
                False
            ),


        "guardrail_decision":

            guardrail_result.get(
                "approved_action",
                ""
            ),


        "action_executed":

            action_executed,


        "recovery_result":

            recovery_result,


        # IMPORTANT
        # Razorpay TEST order ID
        "razorpay_test_order_id":

            razorpay_order_id,


        "revenue_rescued":

            float(
                revenue_rescued
            )

    }


    audit_row = pd.DataFrame(
        [audit_record]
    )


    os.makedirs(
        os.path.dirname(
            AUDIT_PATH
        ),
        exist_ok=True
    )


    # ========================================================
    # EXISTING AUDIT FILE
    # ========================================================

    if os.path.exists(AUDIT_PATH):

        existing_audit = pd.read_csv(
            AUDIT_PATH,
            keep_default_na=False
        )


        # Add Razorpay column if old file
        # doesn't contain it.

        if "razorpay_test_order_id" not in existing_audit.columns:

            existing_audit[
                "razorpay_test_order_id"
            ] = ""


        # Make sure every current column exists.

        for column in audit_record.keys():

            if column not in existing_audit.columns:

                existing_audit[
                    column
                ] = ""


        # Put columns into the correct order.

        existing_audit = existing_audit[
            list(
                audit_record.keys()
            )
        ]


        # Add new record.

        existing_audit = pd.concat(

            [

                existing_audit,

                audit_row

            ],

            ignore_index=True

        )


        # Rewrite CSV.

        existing_audit.to_csv(

            AUDIT_PATH,

            index=False

        )


    else:

        audit_row.to_csv(

            AUDIT_PATH,

            index=False

        )


# ============================================================
# SESSION STATE
# ============================================================

if "selected_transaction" not in st.session_state:

    st.session_state.selected_transaction = None


if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None


if "razorpay_order_id" not in st.session_state:

    st.session_state.razorpay_order_id = ""


# ============================================================
# ML PREDICTION
# ============================================================

def predict_recovery_probability(transaction):

    if recovery_model is None:

        return 0.0


    input_df = pd.DataFrame(

        [

            {

                "amount":
                    transaction["amount"],

                "payment_method":
                    transaction["payment_method"],

                "customer_success_rate":
                    transaction[
                        "customer_success_rate"
                    ],

                "previous_failed_attempts":
                    transaction[
                        "previous_failed_attempts"
                    ],

                "is_new_device":
                    transaction[
                        "is_new_device"
                    ],

                "failure_reason":
                    transaction[
                        "failure_reason"
                    ]

            }

        ]

    )


    probability = recovery_model.predict_proba(
        input_df
    )[0][1]


    return float(
        probability
    )


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def deterministic_recommendation(transaction, probability):

    failure_reason = str(
        transaction[
            "failure_reason"
        ]
    ).lower()


    attempts = int(
        transaction[
            "previous_failed_attempts"
        ]
    )


    if probability < 0.25:

        action = "DO_NOT_RETRY"

        priority = "LOW"


    elif "expired card" in failure_reason:

        action = "UPDATE_PAYMENT_METHOD"

        priority = "HIGH"


    elif attempts >= 3:

        action = "DO_NOT_RETRY"

        priority = "HIGH"


    elif probability >= 0.75:

        action = "RETRY_NOW"

        priority = "HIGH"


    elif probability >= 0.55:

        action = "RETRY_LATER"

        priority = "MEDIUM"


    else:

        action = "ADDITIONAL_AUTHENTICATION"

        priority = "MEDIUM"


    return {

        "action":
            action,

        "priority":
            priority,

        "reason":
            "Deterministic fallback recommendation.",

        "fallback_used":
            True

    }


# ============================================================
# GEMINI AI DECISION
# ============================================================

def get_gemini_recommendation(

    transaction,

    probability

):


    fallback = deterministic_recommendation(

        transaction,

        probability

    )


    if gemini_client is None:

        return fallback


    prompt = f"""

You are the AI decision engine for RevenueRescue AI.

This is a PAYMENT RECOVERY SIMULATION.

Do not process real money.

Analyze the failed payment and recommend exactly ONE action.

Allowed actions:

DO_NOT_RETRY
UPDATE_PAYMENT_METHOD
RETRY_NOW
RETRY_LATER
ADDITIONAL_AUTHENTICATION

Transaction:

Transaction ID:
{transaction["transaction_id"]}

Amount:
₹{float(transaction["amount"]):.2f}

Payment Method:
{transaction["payment_method"]}

Customer Success Rate:
{transaction["customer_success_rate"]}

Previous Failed Attempts:
{transaction["previous_failed_attempts"]}

New Device:
{transaction["is_new_device"]}

Failure Reason:
{transaction["failure_reason"]}

ML Recovery Probability:
{probability:.4f}

Return the decision in this exact format:

ACTION: <one allowed action>
PRIORITY: <LOW/MEDIUM/HIGH>
REASON: <short reason>

"""


    try:

        response = gemini_client.models.generate_content(

            model="gemini-3.7-flash",

            contents=prompt

        )


        text = response.text.strip()


        action = None
        priority = None
        reason = ""


        for line in text.splitlines():

            line = line.strip()


            if line.startswith("ACTION:"):

                action = line.split(
                    ":",
                    1
                )[1].strip()


            elif line.startswith("PRIORITY:"):

                priority = line.split(
                    ":",
                    1
                )[1].strip()


            elif line.startswith("REASON:"):

                reason = line.split(
                    ":",
                    1
                )[1].strip()


        allowed_actions = [

            "DO_NOT_RETRY",

            "UPDATE_PAYMENT_METHOD",

            "RETRY_NOW",

            "RETRY_LATER",

            "ADDITIONAL_AUTHENTICATION"

        ]


        if action not in allowed_actions:

            return fallback


        if priority not in [

            "LOW",

            "MEDIUM",

            "HIGH"

        ]:

            priority = "MEDIUM"


        return {

            "action":
                action,

            "priority":
                priority,

            "reason":
                reason,

            "fallback_used":
                False

        }


    except Exception:

        return fallback


# ============================================================
# SAFETY GUARDRAILS
# ============================================================

def apply_guardrails(

    transaction,

    ai_result,

    probability

):


    requested_action = ai_result.get(
        "action",
        "DO_NOT_RETRY"
    )


    attempts = int(
        transaction[
            "previous_failed_attempts"
        ]
    )


    failure_reason = str(
        transaction[
            "failure_reason"
        ]
    ).lower()


    # --------------------------------------------------------
    # STOP RULE
    # --------------------------------------------------------

    if attempts >= 3:

        return {

            "approved_action":
                "DO_NOT_RETRY",

            "reason":
                "Maximum retry threshold reached."

        }


    # --------------------------------------------------------
    # EXPIRED CARD
    # --------------------------------------------------------

    if "expired card" in failure_reason:

        return {

            "approved_action":
                "UPDATE_PAYMENT_METHOD",

            "reason":
                "Expired card requires payment method update."

        }


    # --------------------------------------------------------
    # LOW PROBABILITY
    # --------------------------------------------------------

    if probability < 0.20:

        return {

            "approved_action":
                "DO_NOT_RETRY",

            "reason":
                "Recovery probability below safety threshold."

        }


    # --------------------------------------------------------
    # VALIDATE AI ACTION
    # --------------------------------------------------------

    allowed_actions = [

        "DO_NOT_RETRY",

        "UPDATE_PAYMENT_METHOD",

        "RETRY_NOW",

        "RETRY_LATER",

        "ADDITIONAL_AUTHENTICATION"

    ]


    if requested_action not in allowed_actions:

        return {

            "approved_action":
                "DO_NOT_RETRY",

            "reason":
                "AI action failed validation."

        }


    return {

        "approved_action":
            requested_action,

        "reason":
            "Action passed safety guardrails."

    }


# ============================================================
# EXECUTE RECOVERY
# ============================================================

def execute_recovery(

    transaction,

    action,

    ml_probability,

    ai_result,

    guardrail_result

):


    razorpay_order_id = ""

    recovery_result = "NOT_EXECUTED"

    revenue_rescued = 0.0


    # ========================================================
    # DO NOT RETRY
    # ========================================================

    if action == "DO_NOT_RETRY":

        recovery_result = "STOPPED"

        revenue_rescued = 0.0


    # ========================================================
    # UPDATE PAYMENT METHOD
    # ========================================================

    elif action == "UPDATE_PAYMENT_METHOD":

        st.info(
            "Simulating payment method update..."
        )


        # In this demo, the payment method
        # update is simulated.

        order_result = create_razorpay_test_order(
            transaction
        )


        if order_result["success"]:

            razorpay_order_id = (
                order_result["order_id"]
            )

            recovery_result = (
                "TEST_ORDER_CREATED"
            )

            revenue_rescued = float(
                transaction["amount"]
            )

        else:

            recovery_result = (
                "RAZORPAY_TEST_ORDER_FAILED"
            )


    # ========================================================
    # RETRY NOW
    # ========================================================

    elif action == "RETRY_NOW":

        order_result = create_razorpay_test_order(
            transaction
        )


        if order_result["success"]:

            razorpay_order_id = (
                order_result["order_id"]
            )

            recovery_result = (
                "TEST_ORDER_CREATED"
            )

            revenue_rescued = float(
                transaction["amount"]
            )

        else:

            recovery_result = (
                "RAZORPAY_TEST_ORDER_FAILED"
            )


    # ========================================================
    # RETRY LATER
    # ========================================================

    elif action == "RETRY_LATER":

        st.info(
            "Recovery scheduled for later."
        )


        order_result = create_razorpay_test_order(
            transaction
        )


        if order_result["success"]:

            razorpay_order_id = (
                order_result["order_id"]
            )

            recovery_result = (
                "TEST_ORDER_CREATED_SCHEDULED"
            )

            revenue_rescued = float(
                transaction["amount"]
            )

        else:

            recovery_result = (
                "RAZORPAY_TEST_ORDER_FAILED"
            )


    # ========================================================
    # ADDITIONAL AUTHENTICATION
    # ========================================================

    elif action == "ADDITIONAL_AUTHENTICATION":

        st.info(
            "Simulating additional authentication..."
        )


        order_result = create_razorpay_test_order(
            transaction
        )


        if order_result["success"]:

            razorpay_order_id = (
                order_result["order_id"]
            )

            recovery_result = (
                "TEST_ORDER_CREATED_AFTER_AUTH"
            )

            revenue_rescued = float(
                transaction["amount"]
            )

        else:

            recovery_result = (
                "RAZORPAY_TEST_ORDER_FAILED"
            )


    # ========================================================
    # SAVE RAZORPAY ORDER ID
    # ========================================================

    st.session_state.razorpay_order_id = (
        razorpay_order_id
    )


    # ========================================================
    # WRITE AUDIT RECORD
    # ========================================================

    write_audit_record(

        transaction=

            transaction,

        ml_probability=

            ml_probability,

        ai_result=

            ai_result,

        guardrail_result=

            guardrail_result,

        action_executed=

            action,

        recovery_result=

            recovery_result,

        revenue_rescued=

            revenue_rescued,

        razorpay_order_id=

            razorpay_order_id

    )


    return {

        "recovery_result":
            recovery_result,

        "revenue_rescued":
            revenue_rescued,

        "razorpay_order_id":
            razorpay_order_id

    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "💰 RevenueRescue AI"
)

st.subheader(
    "AI-Powered Payment Recovery System"
)


st.warning(
    """
    ⚠️ DEMO / SIMULATION ONLY

    This application does not process real payments.
    Razorpay integration uses TEST MODE only.
    Revenue rescued values shown in the dashboard are simulated.
    """
)


# ============================================================
# RAZORPAY STATUS
# ============================================================

if razorpay_client:

    st.success(
        "🟢 Razorpay Test Mode connected"
    )

else:

    st.warning(
        "🟡 Razorpay Test Mode credentials not configured. "
        "The recovery workflow will still run in simulation mode."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(

    [

        "🤖 Recovery Agent",

        "📊 Analytics",

        "🧾 Audit Trail"

    ]

)


# ============================================================
# TAB 1 — RECOVERY AGENT
# ============================================================

with tab1:

    st.header(
        "AI Payment Recovery Agent"
    )


    if transactions_df.empty:

        st.error(
            "Transaction dataset could not be loaded."
        )

    else:

        failed_df = transactions_df[
            transactions_df["status"].astype(str).str.lower()
            == "failed"
        ].copy()


        if failed_df.empty:

            st.success(
                "No failed transactions found."
            )

        else:

            transaction_options = (

                failed_df[
                    "transaction_id"
                ]
                .astype(str)
                .tolist()

            )


            selected_id = st.selectbox(

                "Select a failed transaction",

                transaction_options

            )


            selected_transaction = (

                failed_df[
                    failed_df[
                        "transaction_id"
                    ].astype(str)
                    == selected_id
                ]
                .iloc[0]
                .copy()

            )


            st.session_state.selected_transaction = (
                selected_transaction
            )


            # ------------------------------------------------
            # TRANSACTION DETAILS
            # ------------------------------------------------

            st.subheader(
                "Transaction Details"
            )


            col1, col2, col3, col4 = st.columns(4)


            col1.metric(

                "Transaction",

                str(
                    selected_transaction[
                        "transaction_id"
                    ]
                )

            )


            col2.metric(

                "Amount",

                f"₹{float(selected_transaction['amount']):,.2f}"

            )


            col3.metric(

                "Payment Method",

                str(
                    selected_transaction[
                        "payment_method"
                    ]
                )

            )


            col4.metric(

                "Failure Reason",

                str(
                    selected_transaction[
                        "failure_reason"
                    ]
                )

            )


            st.write(

                "Previous Failed Attempts:",

                selected_transaction[
                    "previous_failed_attempts"
                ]

            )


            st.write(

                "Customer Success Rate:",

                selected_transaction[
                    "customer_success_rate"
                ]

            )


            # ------------------------------------------------
            # ANALYZE BUTTON
            # ------------------------------------------------

            if st.button(

                "🔍 Analyze Payment with AI",

                use_container_width=True

            ):

                with st.spinner(
                    "Running ML + AI analysis..."
                ):


                    probability = (

                        predict_recovery_probability(

                            selected_transaction

                        )

                    )


                    ai_result = (

                        get_gemini_recommendation(

                            selected_transaction,

                            probability

                        )

                    )


                    guardrail_result = (

                        apply_guardrails(

                            selected_transaction,

                            ai_result,

                            probability

                        )

                    )


                    st.session_state.analysis_result = {

                        "probability":
                            probability,

                        "ai_result":
                            ai_result,

                        "guardrail_result":
                            guardrail_result

                    }


            # ------------------------------------------------
            # SHOW ANALYSIS
            # ------------------------------------------------

            if st.session_state.analysis_result:

                result = (
                    st.session_state.analysis_result
                )


                probability = (
                    result["probability"]
                )


                ai_result = (
                    result["ai_result"]
                )


                guardrail_result = (
                    result["guardrail_result"]
                )


                st.divider()


                st.subheader(
                    "🧠 AI Decision"
                )


                col1, col2, col3 = st.columns(3)


                col1.metric(

                    "ML Recovery Probability",

                    f"{probability * 100:.2f}%"

                )


                col2.metric(

                    "AI Recommendation",

                    ai_result[
                        "action"
                    ]

                )


                col3.metric(

                    "Priority",

                    ai_result[
                        "priority"
                    ]

                )


                if ai_result.get(
                    "fallback_used",
                    False
                ):

                    st.info(
                        "Gemini unavailable or returned an invalid response. "
                        "Deterministic fallback logic was used."
                    )

                else:

                    st.success(
                        "Gemini AI decision generated successfully."
                    )


                st.write(

                    "**AI Reason:**",

                    ai_result.get(
                        "reason",
                        ""
                    )

                )


                st.divider()


                st.subheader(
                    "🛡️ Safety Guardrail"
                )


                st.info(

                    f"Approved Action: "
                    f"**{guardrail_result['approved_action']}**"

                )


                st.write(

                    guardrail_result[
                        "reason"
                    ]

                )


                # ------------------------------------------------
                # EXECUTE RECOVERY
                # ------------------------------------------------

                st.divider()


                approved_action = (

                    guardrail_result[
                        "approved_action"
                    ]

                )


                if st.button(

                    "🚀 Execute Recovery Action",

                    use_container_width=True

                ):

                    with st.spinner(
                        "Executing recovery workflow..."
                    ):

                        execution_result = (

                            execute_recovery(

                                selected_transaction,

                                approved_action,

                                probability,

                                ai_result,

                                guardrail_result

                            )

                        )


                    st.success(
                        "Recovery workflow executed."
                    )


                    # --------------------------------------------
                    # RECOVERY RESULT
                    # --------------------------------------------

                    st.subheader(
                        "Recovery Result"
                    )


                    col1, col2 = st.columns(2)


                    col1.metric(

                        "Recovery Result",

                        execution_result[
                            "recovery_result"
                        ]

                    )


                    col2.metric(

                        "Revenue Rescued "
                        "(Simulation)",

                        f"₹{execution_result['revenue_rescued']:,.2f}"

                    )


                    # --------------------------------------------
                    # RAZORPAY ORDER ID
                    # --------------------------------------------

                    if execution_result[
                        "razorpay_order_id"
                    ]:

                        st.success(

                            "✅ Razorpay Test Mode Order Created"

                        )


                        st.code(

                            execution_result[
                                "razorpay_order_id"
                            ],

                            language="text"

                        )


                        st.caption(

                            "This is a Razorpay TEST MODE order ID. "
                            "No real money was processed."

                        )


                    else:

                        st.info(
                            "No Razorpay Test Mode order was created for this action."
                        )


# ============================================================
# TAB 2 — ANALYTICS
# ============================================================

with tab2:

    st.header(
        "📊 Revenue Recovery Analytics"
    )


    if transactions_df.empty:

        st.warning(
            "Transaction data unavailable."
        )

    else:

        failed_transactions = transactions_df[
            transactions_df["status"].astype(str).str.lower()
            == "failed"
        ]


        total_transactions = len(
            transactions_df
        )


        failed_count = len(
            failed_transactions
        )


        total_value = (
            transactions_df["amount"]
            .astype(float)
            .sum()
        )


        revenue_at_risk = (
            failed_transactions["amount"]
            .astype(float)
            .sum()
        )


        recovered_revenue = (

            failed_transactions[
                failed_transactions[
                    "recovered"
                ].astype(int)
                == 1
            ]["amount"]
            .astype(float)
            .sum()

        )


        unrecovered_revenue = (

            revenue_at_risk
            - recovered_revenue

        )


        recovery_count = len(

            failed_transactions[
                failed_transactions[
                    "recovered"
                ].astype(int)
                == 1
            ]

        )


        recovery_rate = (

            recovery_count
            / failed_count
            * 100

            if failed_count > 0

            else 0

        )


        # ------------------------------------------------
        # METRICS
        # ------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        col1.metric(

            "Total Transactions",

            f"{total_transactions:,}"

        )


        col2.metric(

            "Failed Transactions",

            f"{failed_count:,}"

        )


        col3.metric(

            "Revenue at Risk",

            f"₹{revenue_at_risk:,.2f}"

        )


        col4.metric(

            "Historical Recovery Rate",

            f"{recovery_rate:.2f}%"

        )


        st.divider()


        col1, col2 = st.columns(2)


        col1.metric(

            "Recovered Revenue",

            f"₹{recovered_revenue:,.2f}"

        )


        col2.metric(

            "Unrecovered Revenue",

            f"₹{unrecovered_revenue:,.2f}"

        )


        # ------------------------------------------------
        # FAILURE REASON ANALYSIS
        # ------------------------------------------------

        st.subheader(
            "Failure Reason Analysis"
        )


        reason_analysis = (

            failed_transactions
            .groupby("failure_reason")
            .agg(

                failed_transactions=(
                    "transaction_id",
                    "count"
                ),

                revenue_at_risk=(
                    "amount",
                    "sum"
                ),

                recovered_revenue=(
                    "amount",
                    lambda x:
                        x[
                            failed_transactions.loc[
                                x.index,
                                "recovered"
                            ].astype(int)
                            == 1
                        ].sum()
                )

            )
            .reset_index()

        )


        reason_analysis[
            "recovery_rate"
        ] = (

            reason_analysis[
                "recovered_revenue"
            ]

            /

            reason_analysis[
                "revenue_at_risk"
            ]

            * 100

        )


        st.dataframe(

            reason_analysis,

            use_container_width=True,

            hide_index=True

        )


        # ------------------------------------------------
        # MODEL PERFORMANCE
        # ------------------------------------------------

        st.subheader(
            "🤖 ML Model Performance"
        )


        if model_metrics:

            col1, col2, col3, col4 = st.columns(4)


            col1.metric(

                "Accuracy",

                f"{model_metrics['accuracy']:.4f}"

            )


            col2.metric(

                "Precision",

                f"{model_metrics['precision']:.4f}"

            )


            col3.metric(

                "Recall",

                f"{model_metrics['recall']:.4f}"

            )


            col4.metric(

                "ROC-AUC",

                f"{model_metrics['roc_auc']:.4f}"

            )


            st.write(
                "Confusion Matrix:"
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


            st.dataframe(
                cm_df,
                use_container_width=True
            )


# ============================================================
# TAB 3 — AUDIT TRAIL
# ============================================================

with tab3:

    st.header(
        "🧾 Recovery Audit Trail"
    )


    audit_df = load_audit_trail()


    if audit_df.empty:

        st.info(
            "No recovery actions have been recorded yet."
        )

    else:

        # ------------------------------------------------
        # FORCE RAZORPAY COLUMN
        # ------------------------------------------------

        if "razorpay_test_order_id" not in audit_df.columns:

            audit_df[
                "razorpay_test_order_id"
            ] = ""


        # ------------------------------------------------
        # AUDIT METRICS
        # ------------------------------------------------

        total_actions = len(
            audit_df
        )


        successful_test_orders = (

            audit_df[
                "razorpay_test_order_id"
            ]
            .astype(str)
            .str.startswith(
                "order_"
            )
            .sum()

        )


        total_simulated_revenue = (

            pd.to_numeric(

                audit_df[
                    "revenue_rescued"
                ],

                errors="coerce"

            )
            .fillna(0)
            .sum()

        )


        col1, col2, col3 = st.columns(3)


        col1.metric(

            "Recovery Actions",

            f"{total_actions:,}"

        )


        col2.metric(

            "Razorpay Test Orders",

            f"{successful_test_orders:,}"

        )


        col3.metric(

            "Simulated Revenue Rescued",

            f"₹{total_simulated_revenue:,.2f}"

        )


        st.divider()


        # ------------------------------------------------
        # DISPLAY AUDIT TRAIL
        # ------------------------------------------------

        st.dataframe(

            audit_df,

            use_container_width=True,

            hide_index=True

        )


        # ------------------------------------------------
        # RAZORPAY ORDER IDS
        # ------------------------------------------------

        st.subheader(
            "Razorpay Test Mode Orders"
        )


        razorpay_orders = audit_df[

            audit_df[
                "razorpay_test_order_id"
            ]
            .astype(str)
            .str.startswith(
                "order_"
            )

        ]


        if razorpay_orders.empty:

            st.info(
                "No Razorpay Test Mode order IDs recorded yet."
            )

        else:

            display_columns = [

                "timestamp",

                "transaction_id",

                "amount",

                "action_executed",

                "recovery_result",

                "razorpay_test_order_id",

                "revenue_rescued"

            ]


            available_columns = [

                column

                for column in display_columns

                if column in razorpay_orders.columns

            ]


            st.dataframe(

                razorpay_orders[
                    available_columns
                ],

                use_container_width=True,

                hide_index=True

            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(

    """
    RevenueRescue AI | Razorpay AI Buildathon 2026

    AI-powered payment recovery simulation using
    Random Forest + Gemini + deterministic safety guardrails
    + Razorpay Test Mode.

    No real payments are processed.
    """

)

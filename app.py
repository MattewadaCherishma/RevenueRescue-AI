import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RevenueRescue AI",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# Deployment-safe: works locally and on Streamlit Cloud
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_DIR, "data", "transactions.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "recovery_model.pkl"
)

BATCH_PATH = os.path.join(
    PROJECT_DIR, "outputs", "batch_results.csv"
)

AUDIT_PATH = os.path.join(
    PROJECT_DIR, "outputs", "audit_trail.csv"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

load_dotenv()

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")


client = None

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        client = None


# ============================================================
# LOAD DATA / MODEL
# ============================================================

@st.cache_data
def load_transactions():
    return pd.read_csv(
        DATA_PATH,
        keep_default_na=False
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_batch_results():
    if os.path.exists(BATCH_PATH):
        return pd.read_csv(BATCH_PATH)
    return pd.DataFrame()


def load_audit_trail():
    if os.path.exists(AUDIT_PATH):
        return pd.read_csv(AUDIT_PATH)
    return pd.DataFrame()


transactions_df = load_transactions()
pipeline = load_model()
batch_results_df = load_batch_results()


# ============================================================
# AUDIT LOGGING
# ============================================================

def write_audit_record(
    transaction,
    ml_probability,
    ai_result,
    guardrail_result,
    action_executed,
    recovery_result,
    revenue_rescued
):

    audit_record = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "transaction_id": transaction["transaction_id"],

        "amount": float(transaction["amount"]),

        "failure_reason": transaction["failure_reason"],

        "previous_failed_attempts": int(
            transaction["previous_failed_attempts"]
        ),

        "ml_recovery_probability": float(
            ml_probability
        ),

        "ai_recommendation": ai_result.get(
            "action", ""
        ),

        "ai_priority": ai_result.get(
            "priority", ""
        ),

        "ai_fallback_used": ai_result.get(
            "fallback_used", False
        ),

        "guardrail_decision": guardrail_result.get(
            "approved_action", ""
        ),

        "action_executed": action_executed,

        "recovery_result": recovery_result,

        "revenue_rescued": float(
            revenue_rescued
        )
    }

    audit_row = pd.DataFrame([audit_record])

    os.makedirs(
        os.path.dirname(AUDIT_PATH),
        exist_ok=True
    )

    if os.path.exists(AUDIT_PATH):

        audit_row.to_csv(
            AUDIT_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        audit_row.to_csv(
            AUDIT_PATH,
            mode="w",
            header=True,
            index=False
        )


# ============================================================
# SESSION STATE
# ============================================================

default_states = {
    "selected_transaction": None,
    "payment_method_updated": False,
    "payment_retry_completed": False,
    "authentication_completed": False,
    "recovery_success": False,
    "last_transaction_id": None,
    "ai_results": None
}

for key, value in default_states.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# ML PREDICTION
# ============================================================

def predict_recovery_probability(transaction):

    features = [
        "amount",
        "payment_method",
        "customer_success_rate",
        "previous_failed_attempts",
        "is_new_device",
        "failure_reason"
    ]

    input_df = pd.DataFrame(
        [transaction[features].to_dict()]
    )

    probability = pipeline.predict_proba(
        input_df
    )[0][1]

    return float(probability)


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def deterministic_fallback(
    transaction,
    ml_probability
):

    failure_reason = transaction[
        "failure_reason"
    ]

    previous_failed_attempts = int(
        transaction[
            "previous_failed_attempts"
        ]
    )

    if previous_failed_attempts >= 3:

        return {
            "action": "DO_NOT_RETRY",
            "priority": "HIGH",
            "reason": "Maximum retry limit reached.",
            "ai_status":
                "⚠️ Gemini temporarily unavailable. "
                "Deterministic fallback used.",
            "fallback_used": True
        }

    if failure_reason == "Expired Card":

        return {
            "action":
                "UPDATE_PAYMENT_METHOD",

            "priority": "HIGH",

            "reason":
                "Expired card requires a new payment method.",

            "ai_status":
                "⚠️ Gemini temporarily unavailable. "
                "Deterministic fallback used.",

            "fallback_used": True
        }

    if failure_reason == "Authentication Failed":

        return {
            "action":
                "ADDITIONAL_AUTHENTICATION",

            "priority": "HIGH",

            "reason":
                "Additional customer authentication is required.",

            "ai_status":
                "⚠️ Gemini temporarily unavailable. "
                "Deterministic fallback used.",

            "fallback_used": True
        }

    if failure_reason == "Network Error":

        return {
            "action":
                "RETRY_NOW",

            "priority": "MEDIUM",

            "reason":
                "Network errors can often be resolved by retrying the payment.",

            "ai_status":
                "⚠️ Gemini temporarily unavailable. "
                "Deterministic fallback used.",

            "fallback_used": True
        }

    if failure_reason == "Bank Timeout":

        return {
            "action":
                "RETRY_NOW",

            "priority": "MEDIUM",

            "reason":
                "Bank timeout may be temporary, so a retry is appropriate.",

            "ai_status":
                "⚠️ Gemini temporarily unavailable. "
                "Deterministic fallback used.",

            "fallback_used": True
        }

    if failure_reason == "Insufficient Funds":

        return {
            "action":
                "RETRY_LATER",

            "priority": "MEDIUM",

            "reason":
                "Insufficient funds may require waiting before retrying.",

            "ai_status":
                "⚠️ Gemini temporarily unavailable. "
                "Deterministic fallback used.",

            "fallback_used": True
        }

    return {
        "action":
            "DO_NOT_RETRY",

        "priority":
            "LOW",

        "reason":
            "No safe recovery action could be determined.",

        "ai_status":
            "⚠️ Gemini temporarily unavailable. "
            "Deterministic fallback used.",

        "fallback_used": True
    }


# ============================================================
# GEMINI AI DECISION
# ============================================================

def get_ai_recommendation(
    transaction,
    ml_probability
):

    if client is None:

        return deterministic_fallback(
            transaction,
            ml_probability
        )

    prompt = f"""
You are the AI decision engine for RevenueRescue AI,
a payment recovery system.

Analyze the failed payment below.

Transaction ID:
{transaction["transaction_id"]}

Amount:
₹{float(transaction["amount"]):.2f}

Payment Method:
{transaction["payment_method"]}

Customer Success Rate:
{float(transaction["customer_success_rate"]):.2f}

Previous Failed Attempts:
{int(transaction["previous_failed_attempts"])}

New Device:
{transaction["is_new_device"]}

Failure Reason:
{transaction["failure_reason"]}

Machine Learning Recovery Probability:
{ml_probability:.2%}

Choose exactly ONE action from:

RETRY_NOW
RETRY_LATER
UPDATE_PAYMENT_METHOD
ADDITIONAL_AUTHENTICATION
DO_NOT_RETRY

Safety requirements:

- Never recommend retry if previous_failed_attempts >= 3.
- Expired Card should use UPDATE_PAYMENT_METHOD.
- Authentication Failed should use ADDITIONAL_AUTHENTICATION.
- Use RETRY_NOW for temporary network/bank failures.
- Use RETRY_LATER for insufficient funds.
- Use DO_NOT_RETRY when recovery is unsafe.

Return exactly this format:

ACTION: <action>
PRIORITY: <HIGH/MEDIUM/LOW>
REASON: <short reason>
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )

        text = response.text.strip()

        action = "DO_NOT_RETRY"
        priority = "LOW"
        reason = "Gemini response could not be parsed."

        for line in text.splitlines():

            line = line.strip()

            if line.startswith("ACTION:"):
                action = line.replace(
                    "ACTION:",
                    ""
                ).strip()

            elif line.startswith("PRIORITY:"):
                priority = line.replace(
                    "PRIORITY:",
                    ""
                ).strip()

            elif line.startswith("REASON:"):
                reason = line.replace(
                    "REASON:",
                    ""
                ).strip()

        valid_actions = [
            "RETRY_NOW",
            "RETRY_LATER",
            "UPDATE_PAYMENT_METHOD",
            "ADDITIONAL_AUTHENTICATION",
            "DO_NOT_RETRY"
        ]

        if action not in valid_actions:

            return deterministic_fallback(
                transaction,
                ml_probability
            )

        return {
            "action": action,
            "priority": priority,
            "reason": reason,
            "ai_status":
                "🤖 Gemini AI recommendation generated.",
            "fallback_used": False
        }

    except Exception:

        return deterministic_fallback(
            transaction,
            ml_probability
        )


# ============================================================
# SAFETY GUARDRAILS
# ============================================================

def apply_safety_guardrails(
    transaction,
    ai_action
):

    failure_reason = transaction[
        "failure_reason"
    ]

    previous_failed_attempts = int(
        transaction[
            "previous_failed_attempts"
        ]
    )

    if previous_failed_attempts >= 3:

        return {
            "approved_action":
                "DO_NOT_RETRY",

            "guardrail_reason":
                "Maximum retry limit reached."
        }

    if failure_reason == "Expired Card":

        return {
            "approved_action":
                "UPDATE_PAYMENT_METHOD",

            "guardrail_reason":
                "Expired card requires a new payment method."
        }

    if failure_reason == "Authentication Failed":

        return {
            "approved_action":
                "ADDITIONAL_AUTHENTICATION",

            "guardrail_reason":
                "Additional authentication is required."
        }

    return {
        "approved_action":
            ai_action,

        "guardrail_reason":
            "AI recommendation passed the safety checks."
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "💳 RevenueRescue AI"
)

st.subheader(
    "AI-Powered Payment Recovery System"
)

st.write(
    """
RevenueRescue AI identifies failed payments, predicts
recovery probability using machine learning, uses Gemini AI
to recommend recovery actions, applies deterministic safety
guardrails, and executes a bounded recovery workflow with
an auditable outcome.
"""
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "⚡ Recovery Agent",
        "📊 Revenue Analytics",
        "📋 Audit Trail"
    ]
)


# ============================================================
# RECOVERY AGENT
# ============================================================

with tab1:

    failed_df = transactions_df[
        transactions_df["status"] == "Failed"
    ].copy()

    st.markdown(
        "### Select a Failed Transaction"
    )

    transaction_options = failed_df[
        "transaction_id"
    ].tolist()

    selected_transaction_id = st.selectbox(
        "Transaction ID",
        transaction_options
    )

    if (
        st.session_state.last_transaction_id
        != selected_transaction_id
    ):

        st.session_state.payment_method_updated = False
        st.session_state.payment_retry_completed = False
        st.session_state.authentication_completed = False
        st.session_state.recovery_success = False
        st.session_state.ai_results = None

        st.session_state.last_transaction_id = (
            selected_transaction_id
        )

    transaction = failed_df[
        failed_df["transaction_id"]
        == selected_transaction_id
    ].iloc[0]


    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    st.markdown(
        "### Transaction Details"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Amount",
        f"₹{float(transaction['amount']):,.2f}"
    )

    col2.metric(
        "Failure Reason",
        transaction["failure_reason"]
    )

    col3.metric(
        "Previous Failures",
        int(transaction["previous_failed_attempts"])
    )

    col4.metric(
        "Payment Method",
        transaction["payment_method"]
    )


    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    ml_probability = predict_recovery_probability(
        transaction
    )

    st.markdown(
        "### 🤖 AI Decision Engine"
    )

    st.metric(
        "ML Recovery Probability",
        f"{ml_probability:.2%}"
    )


    # --------------------------------------------------------
    # AI DECISION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🧠 Analyze Payment with AI",
        type="primary"
    ):

        ai_result = get_ai_recommendation(
            transaction,
            ml_probability
        )

        guardrail_result = apply_safety_guardrails(
            transaction,
            ai_result["action"]
        )

        st.session_state.ai_results = {
            "ai": ai_result,
            "guardrail": guardrail_result
        }


    # --------------------------------------------------------
    # DISPLAY AI DECISION
    # --------------------------------------------------------

    if st.session_state.ai_results:

        ai_result = st.session_state.ai_results[
            "ai"
        ]

        guardrail_result = st.session_state.ai_results[
            "guardrail"
        ]

        st.info(
            ai_result["ai_status"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Recommended Action",
            ai_result["action"]
        )

        col2.metric(
            "Priority",
            ai_result["priority"]
        )

        col3.metric(
            "Guardrail Decision",
            guardrail_result[
                "approved_action"
            ]
        )

        st.write(
            f"**AI Reason:** {ai_result['reason']}"
        )

        st.write(
            f"**Safety Decision:** "
            f"{guardrail_result['guardrail_reason']}"
        )


        approved_action = guardrail_result[
            "approved_action"
        ]


        # ====================================================
        # DO NOT RETRY
        # ====================================================

        if approved_action == "DO_NOT_RETRY":

            st.warning(
                "🚫 Recovery stopped by safety guardrail. "
                "No payment retry will be executed."
            )


        # ====================================================
        # UPDATE PAYMENT METHOD
        # ====================================================

        elif approved_action == "UPDATE_PAYMENT_METHOD":

            st.markdown(
                "### 🔄 Recovery Workflow"
            )

            if not st.session_state.payment_method_updated:

                if st.button(
                    "💳 Simulate Payment Method Update"
                ):

                    st.session_state.payment_method_updated = True
                    st.rerun()

            else:

                st.success(
                    "✅ Payment method successfully updated "
                    "(simulation)."
                )

                if not st.session_state.payment_retry_completed:

                    if st.button(
                        "🔁 Simulate Payment Retry"
                    ):

                        st.session_state.payment_retry_completed = True
                        st.session_state.recovery_success = True

                        write_audit_record(
                            transaction,
                            ml_probability,
                            ai_result,
                            guardrail_result,
                            "PAYMENT_METHOD_UPDATE → PAYMENT_RETRY",
                            "SUCCESS",
                            float(transaction["amount"])
                        )

                        st.rerun()

                else:

                    st.success(
                        "🎉 Payment retry successful!"
                    )

                    st.metric(
                        "Revenue Rescued",
                        f"₹{float(transaction['amount']):,.2f}"
                    )


        # ====================================================
        # RETRY NOW
        # ====================================================

        elif approved_action == "RETRY_NOW":

            st.markdown(
                "### 🔄 Recovery Workflow"
            )

            if not st.session_state.payment_retry_completed:

                if st.button(
                    "🔁 Simulate Payment Retry"
                ):

                    st.session_state.payment_retry_completed = True
                    st.session_state.recovery_success = True

                    write_audit_record(
                        transaction,
                        ml_probability,
                        ai_result,
                        guardrail_result,
                        "PAYMENT_RETRY",
                        "SUCCESS",
                        float(transaction["amount"])
                    )

                    st.rerun()

            else:

                st.success(
                    "🎉 Payment retry successful!"
                )

                st.metric(
                    "Revenue Rescued",
                    f"₹{float(transaction['amount']):,.2f}"
                )


        # ====================================================
        # RETRY LATER
        # ====================================================

        elif approved_action == "RETRY_LATER":

            st.markdown(
                "### ⏰ Recovery Workflow"
            )

            if not st.session_state.payment_retry_completed:

                if st.button(
                    "⏰ Simulate Scheduled Retry"
                ):

                    st.session_state.payment_retry_completed = True
                    st.session_state.recovery_success = True

                    write_audit_record(
                        transaction,
                        ml_probability,
                        ai_result,
                        guardrail_result,
                        "SCHEDULED_PAYMENT_RETRY",
                        "SUCCESS",
                        float(transaction["amount"])
                    )

                    st.rerun()

            else:

                st.success(
                    "🎉 Scheduled payment retry successful!"
                )

                st.metric(
                    "Revenue Rescued",
                    f"₹{float(transaction['amount']):,.2f}"
                )


        # ====================================================
        # ADDITIONAL AUTHENTICATION
        # ====================================================

        elif approved_action == "ADDITIONAL_AUTHENTICATION":

            st.markdown(
                "### 🔐 Recovery Workflow"
            )

            if not st.session_state.authentication_completed:

                if st.button(
                    "🔐 Simulate Customer Authentication"
                ):

                    st.session_state.authentication_completed = True
                    st.rerun()

            else:

                st.success(
                    "✅ Customer authentication completed "
                    "(simulation)."
                )

                if not st.session_state.payment_retry_completed:

                    if st.button(
                        "🔁 Simulate Payment Retry"
                    ):

                        st.session_state.payment_retry_completed = True
                        st.session_state.recovery_success = True

                        write_audit_record(
                            transaction,
                            ml_probability,
                            ai_result,
                            guardrail_result,
                            "AUTHENTICATION → PAYMENT_RETRY",
                            "SUCCESS",
                            float(transaction["amount"])
                        )

                        st.rerun()

                else:

                    st.success(
                        "🎉 Payment retry successful!"
                    )

                    st.metric(
                        "Revenue Rescued",
                        f"₹{float(transaction['amount']):,.2f}"
                    )


    st.divider()

    st.caption(
        "⚠️ Simulation only. No real payment, card details, "
        "or financial transaction was processed."
    )


# ============================================================
# REVENUE ANALYTICS
# ============================================================

with tab2:

    st.markdown(
        "### 📊 Revenue Analytics"
    )

    total_transactions = len(
        transactions_df
    )

    failed_transactions = len(
        failed_df
    )

    total_value = transactions_df[
        "amount"
    ].sum()

    failed_value = failed_df[
        "amount"
    ].sum()

    recovered_failed = failed_df[
        failed_df["recovered"] == True
    ]

    recovered_revenue = recovered_failed[
        "amount"
    ].sum()

    recovery_rate = (
        len(recovered_failed)
        / failed_transactions
        if failed_transactions > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Failed Payments",
        f"{failed_transactions:,}"
    )

    col3.metric(
        "Revenue at Risk",
        f"₹{failed_value:,.2f}"
    )

    col4.metric(
        "Actual Dataset Recovery Rate",
        f"{recovery_rate:.2%}"
    )


    st.markdown(
        "### 💰 Dataset Recovery Summary"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recovered Revenue",
        f"₹{recovered_revenue:,.2f}"
    )

    col2.metric(
        "Unrecovered Revenue",
        f"₹{failed_value - recovered_revenue:,.2f}"
    )

    col3.metric(
        "Recovered Failed Payments",
        f"{len(recovered_failed):,}"
    )


    # --------------------------------------------------------
    # SIMULATED BATCH RESULTS
    # --------------------------------------------------------

    if not batch_results_df.empty:

        st.markdown(
            "### ⚡ Simulated AI Recovery Batch"
        )

        simulated_revenue = batch_results_df[
            batch_results_df[
                "simulated_recovered"
            ] == True
        ]["amount"].sum()

        simulated_count = batch_results_df[
            "simulated_recovered"
        ].sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Simulated Recovered Payments",
            f"{int(simulated_count):,}"
        )

        col2.metric(
            "Simulated Revenue Rescued",
            f"₹{simulated_revenue:,.2f}"
        )

        col3.metric(
            "Failed Payments Processed",
            f"{len(batch_results_df):,}"
        )

        st.caption(
            "⚠️ Batch recovery figures are simulated "
            "using bounded recovery probabilities and "
            "do not represent real financial transactions."
        )


# ============================================================
# AUDIT TRAIL
# ============================================================

with tab3:

    st.markdown(
        "### 📋 Recovery Audit Trail"
    )

    st.write(
        """
Every completed recovery workflow is recorded with its
AI decision, safety decision, action, outcome, and rescued
revenue.
"""
    )

    audit_df = load_audit_trail()

    if audit_df.empty:

        st.info(
            "No completed recovery workflows have been recorded yet."
        )

    else:

        successful_audits = audit_df[
            audit_df["recovery_result"]
            == "SUCCESS"
        ]

        total_events = len(
            audit_df
        )

        total_rescued = audit_df[
            "revenue_rescued"
        ].sum()

        successful_count = len(
            successful_audits
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Recovery Events",
            f"{total_events:,}"
        )

        col2.metric(
            "Revenue Rescued",
            f"₹{total_rescued:,.2f}"
        )

        col3.metric(
            "Successful Recoveries",
            f"{successful_count:,}"
        )


        st.markdown(
            "### 🧾 Recovery Records"
        )

        display_audit = audit_df.sort_values(
            "timestamp",
            ascending=False
        )

        st.dataframe(
            display_audit,
            use_container_width=True
        )


        csv_data = display_audit.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Audit Trail CSV",
            data=csv_data,
            file_name="revenuerescue_audit_trail.csv",
            mime="text/csv"
        )


        st.caption(
            "⚠️ Recovery actions shown here are "
            "simulation-only and do not represent "
            "real payment transactions."
        )

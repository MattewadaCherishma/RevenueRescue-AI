# 💰 RevenueRescue AI

### AI-Powered Payment Recovery System

RevenueRescue AI is an AI-powered payment recovery system built for the **Razorpay AI Buildathon 2026**.

It identifies failed payments, predicts the probability of successful recovery using Machine Learning, uses an AI decision layer to recommend the best recovery action, applies deterministic safety guardrails, executes the recovery workflow through **Razorpay Test Mode**, and maintains an auditable record of every action.

---

## 🚀 Live Demo

👉 **[Open RevenueRescue AI Live App](https://revenuerescue-ai.streamlit.app/)**

> ⚠️ **Demo Disclaimer:** This application uses Razorpay **Test Mode** only. No real payments or real money are processed. Revenue rescued values displayed by the application are simulated for demonstration and evaluation.

---

## 🎯 Problem Statement

Failed payments create direct revenue loss for businesses.

A failed transaction does not always mean lost revenue. Different failure reasons require different recovery strategies. Blindly retrying every failed payment can result in unnecessary retries, poor customer experience, and ineffective recovery.

RevenueRescue AI addresses this problem by intelligently deciding:

* Whether a payment should be retried
* When it should be retried
* Whether the customer should update their payment method
* Whether additional authentication should be requested
* When the system should stop retrying

---

## 💡 Solution

RevenueRescue AI combines:

**Machine Learning + Generative AI + Safety Guardrails + Razorpay Test Mode + Audit Trail**

The system first estimates the probability of payment recovery using a Random Forest model.

The AI decision layer then recommends an appropriate recovery action.

A deterministic safety layer validates the recommendation before execution.

Finally, the system executes the recovery workflow using Razorpay Test Mode and records the action in the audit trail.

---

## 🔄 System Workflow

```text
Failed Payment
      ↓
Transaction Analysis
      ↓
Random Forest Recovery Prediction
      ↓
Gemini AI Decision Engine
      ↓
Safety & Policy Guardrails
      ↓
Approved Recovery Action
      ↓
Razorpay Test Mode
      ↓
Audit Trail
      ↓
Recovery Analytics
```

---

## 🤖 AI & Machine Learning

### Machine Learning Model

A **Random Forest Classifier** is used to predict the probability that a failed payment can be recovered.

### Features Used

* Transaction amount
* Payment method
* Customer success rate
* Previous failed attempts
* New device indicator
* Failure reason

### Model Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.6963 |
| Precision | 0.7131 |
| Recall    | 0.7275 |
| ROC-AUC   | 0.7463 |

The model achieved a **0.7463 ROC-AUC on a held-out test set**.

---

## 🧠 Gemini AI Decision Layer

Gemini analyzes transaction context and recommends one of the following bounded actions:

```text
DO_NOT_RETRY
UPDATE_PAYMENT_METHOD
RETRY_NOW
RETRY_LATER
ADDITIONAL_AUTHENTICATION
```

If the AI service is unavailable or produces an invalid recommendation, the system uses a deterministic fallback decision engine.

---

## 🛡️ Safety & Guardrails

RevenueRescue AI does not allow unrestricted payment retries.

The system applies deterministic safety rules including:

* Maximum retry threshold
* Low recovery-probability stopping rule
* Expired-card handling
* Allowed-action validation
* AI fallback mechanism
* Explicit Test Mode execution

These guardrails ensure that AI recommendations remain bounded and auditable.

---

## 💳 Razorpay Integration

The application integrates with **Razorpay Test Mode** to demonstrate the recovery execution workflow.

When an approved recovery action is executed, the system can create a Razorpay Test Mode order and record the resulting order ID in the audit trail.

Example:

```text
Razorpay Test Order ID:
order_XXXXXXXXXXXX
```

> Razorpay Test Mode is used only for demonstration. Creating a Test Mode order does not represent a successful real-world payment.

---

## 🧾 Audit Trail

Every recovery action is recorded for traceability.

The audit trail includes:

* Timestamp
* Transaction ID
* Transaction amount
* Failure reason
* Previous failed attempts
* ML recovery probability
* AI recommendation
* AI priority
* Whether fallback was used
* Guardrail decision
* Executed action
* Recovery result
* Razorpay Test Mode Order ID
* Simulated revenue rescued

---

## 📊 Dataset & Analytics

The project uses a dataset containing:

* **10,000 transactions**
* **3,209 failed transactions**
* **₹47.64 lakh revenue at risk**
* **1,726 historically recovered failed payments**
* **53.79% historical recovery rate**

### Historical Revenue

| Metric                  |       Value |
| ----------------------- | ----------: |
| Total Transaction Value | ₹1.51 crore |
| Revenue at Risk         | ₹47.64 lakh |
| Recovered Revenue       | ₹25.28 lakh |
| Unrecovered Revenue     | ₹22.36 lakh |

These figures represent the historical dataset used for analysis.

---

## 📈 Failure Reason Analysis

The system analyzes recovery behavior across different payment failure reasons, including:

* Authentication Failed
* Bank Timeout
* Expired Card
* Insufficient Funds
* Network Error

This helps identify which failure types are more suitable for recovery interventions.

---

## 🛠️ Technology Stack

### Programming & Data

* Python
* Pandas
* NumPy
* Scikit-learn

### AI

* Google Gemini

### Payment Integration

* Razorpay Test Mode

### Application

* Streamlit

### Analytics

* Power BI
* DAX

### Development

* Jupyter Notebook
* VS Code
* GitHub

---

## 📁 Project Structure

```text
RevenueRescue-AI/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── transactions.csv
│
├── models/
│   └── recovery_model.pkl
│
└── outputs/
    ├── audit_trail.csv
    └── batch_results.csv
```

---

## 🚀 Running Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Environment Variables

API credentials are **not stored in the repository**.

For local development, configure:

```text
GEMINI_API_KEY
RAZORPAY_KEY_ID
RAZORPAY_KEY_SECRET
```

For Streamlit Cloud deployment, these credentials are configured using Streamlit Secrets.

> Never commit API keys or `.env` files to GitHub.

---

## 🎬 Demo Flow

The application demonstrates the following workflow:

1. Select a failed transaction.
2. Analyze the transaction using ML + AI.
3. Generate a recovery recommendation.
4. Apply safety guardrails.
5. Execute the approved recovery action.
6. Create a Razorpay Test Mode order when applicable.
7. Record the Razorpay Test Order ID.
8. View the complete action in the Audit Trail.
9. Review recovery analytics.

---

## ⚠️ Disclaimer

RevenueRescue AI is a **prototype/demo for the Razorpay AI Buildathon 2026**.

* Razorpay integration uses **Test Mode only**.
* No real payments are processed.
* No real money is transferred.
* Revenue rescued values in the recovery workflow are simulated.
* The ML model is a prototype and should not be treated as a production payment decision system.

---

## 👩‍💻 Project

**RevenueRescue AI — AI-Powered Payment Recovery System**

Built for the **Razorpay AI Buildathon 2026**.

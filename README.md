# Agentic AI Customer Support System

A production-style, multi-agent customer support platform built with Python, Streamlit, Gemini, RAG, and LangGraph orchestration.

This system is designed as a decision-making support agent, not a generic chatbot.  
It classifies user issues, retrieves policy/context, validates live state from tools, reasons over contradictions, takes actions (refund/unblock), and produces transparent, actionable responses.

---

## Why This Project Is Different

Most support demos stop at: *"retrieve similar ticket -> generate answer"*.  
This project implements: **retrieve -> verify -> reason -> act -> explain**.

Key differentiators:

- LangGraph state-machine orchestration for explicit multi-step reasoning.
- Deterministic policy checks after tool execution (not LLM-only decisions).
- Contradiction handling between user claim and verified system state.
- Confidence-based routing with evidence-aware thresholds.
- Real tool side-effects on JSON data (refund/unblock updates state).
- Transparent live trace of every step in the Streamlit UI.

---

## System Architecture

### Core Runtime

- **Frontend:** Streamlit app with:
  - user login flow
  - chat + live execution stream
  - admin observability dashboard
- **Orchestration:** LangGraph (`StateGraph`) pipeline
- **LLM:** Gemini (optional, via `GEMINI_API_KEY`) with deterministic fallbacks
- **RAG:** SentenceTransformer embeddings + FAISS vector index
- **State/Data:** JSON datasets in `data/`

### High-Level Flow

1. Prepare context (query + remembered order_id if available)
2. Classify intent
3. Retrieve relevant knowledge and similar historical tickets
4. Compute confidence and route decision
5. Plan tools using deterministic planner rules
6. Execute planned tools
7. Run mandatory post-check reasoning rules
8. Generate final response (tool-grounded + user-actionable)
9. Persist ticket + trace and optionally update vector memory

---

## Agent Design (What Each Agent Does)

### 1) Intent Classifier Agent (`agents/classifier.py`)

Purpose:
- Classifies support issue into one of:
  - `payment_issue`
  - `refund`
  - `delivery_delay`
  - `account_issue`
  - `order_cancel`

How:
- First tries Gemini JSON classification.
- Falls back to semantic prototype similarity using embeddings.
- Emits confidence score used downstream for routing.

### 2) Retrieval Agent (`agents/retrieval.py`)

Purpose:
- Retrieves top-k relevant support knowledge + past tickets.

How:
- Uses FAISS semantic search over:
  - `knowledge_base.json`
  - prior support tickets
- Returns ranked snippets with metadata for evidence.

### 3) Planning Agent (`agents/planner.py`)

Purpose:
- Converts intent + confidence + context into executable tool plan.

How:
- Uses deterministic support rules:
  - payment -> check transaction/order state
  - account -> check account status (do not ask irrelevant order_id)
  - delivery -> check order status before escalation
- Avoids asking users for data already present in system memory/db.

### 4) Action/Orchestrator Agent (`agents/langgraph_action_agent.py`)

Purpose:
- Central decision-making state machine.

How:
- Implements LangGraph nodes:
  - `prepare_context`
  - `classify_intent`
  - `retrieve_rag`
  - `route_confidence`
  - `plan_tools`
  - `execute_plan_tools`
  - `post_checks`
  - `generate_and_finalize`

Special logic implemented:
- Mandatory deterministic post-checks by intent.
- Contradiction handling (e.g., refund expectation vs confirmed order).
- Confidence correction when DB signal is strong.
- Guardrails to prevent generic/non-actionable final responses.

### 5) Response Generator Agent (`agents/response_generator.py`)

Purpose:
- Produces final user-facing response in structured format:
  - explanation
  - actions taken
  - next steps
  - final message

How:
- Prefers deterministic, tool-grounded responses when tool signals exist.
- Uses Gemini only as optional stylistic upgrader.
- Includes output sanitization to avoid noisy historical dump text.

---

## Tool Layer (Operational APIs)

### Payment Tools (`tools/payment_tools.py`)

- `check_payment_status(order_id)`
  - validates order/payment state from datasets
- `initiate_refund(order_id, reason)`
  - updates order payment state
  - appends refund transaction record

### Account Tools (`tools/account_tools.py`)

- `get_user_account_status(user_id)`
- `unblock_account(user_id)`

### Order Tools (`tools/order_tools.py`)

- `get_order_details(order_id)`
- `get_user_orders(user_id)`

All tools are integrated into agent trace for transparency.

---

## Reasoning & Decision Policies Implemented

### Confidence Routing (`utils/confidence.py`)

- `final_confidence = 0.6 * retrieval_score + 0.4 * intent_score`
- Routing:
  - `> 0.65` -> `auto_resolve`
  - `0.40 - 0.65` -> `ask_clarification`
  - `< 0.40` -> `escalate_admin`

### Deterministic Post-Tool Policies

- Payment mismatch (paid + missing order confirmation) -> refund path
- Refund contradiction (order already confirmed) -> correct user expectation
- Shipped + not received -> actionable wait window (e.g., ~1 day) then escalation path
- Account blocked -> unblock action
- Delivery delivered-but-user-disagrees -> targeted clarification for proof trail

---

## Streamlit Experience

### User View

- Chat UI with:
  - quick order selector
  - real-time live reasoning steps
  - final resolution + decision badge
- User ticket isolation (users only see their own history)

### Admin View

- Dashboard for:
  - ticket filters
  - confidence and decision visibility
  - full step-by-step execution trace

---

## Project Structure

```text
project/
├── agents/
│   ├── classifier.py
│   ├── retrieval.py
│   ├── planner.py
│   ├── response_generator.py
│   ├── langgraph_action_agent.py
│   └── action_agent.py
├── tools/
│   ├── order_tools.py
│   ├── payment_tools.py
│   └── account_tools.py
├── memory/
│   └── memory_store.py
├── utils/
│   ├── confidence.py
│   ├── embeddings.py
│   ├── logger.py
│   └── vector_store.py
├── data/
│   ├── users.json
│   ├── orders.json
│   ├── transactions.json
│   ├── support_tickets.json
│   └── knowledge_base.json
├── app.py
├── requirements.txt
└── README.md
```

---

## Example Behaviors

- **"Money deducted but order not received"**
  - checks payment/order state
  - decides refund vs reconciliation wait using policy window
- **"Account blocked"**
  - checks account status
  - triggers unblock when blocked
- **"Order not received, status shipped"**
  - confirms in-transit state
  - gives clear wait-and-escalate guidance
- **"Refund expected but order confirmed"**
  - identifies contradiction
  - corrects expectation and suggests proper return/cancel flow

---

## Technical Highlights (Interview Ready)

- LangGraph-based explicit agent graph
- Hybrid LLM + deterministic policy architecture
- RAG with FAISS + sentence embeddings
- Confidence-aware autonomous routing
- Tool-grounded explainable responses
- Stateful memory + user context carry-forward
- Admin observability with full traceability

---

## Notes

- This project is intentionally engineered as a **customer support intelligence system** with operational decision rules, not a general conversational bot.
- `.env` is excluded from git. Use `.env.example` as the template.

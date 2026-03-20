from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.langgraph_action_agent import LangGraphSupportActionAgent
from agents.classifier import IntentClassifier
from agents.planner import Planner
from agents.retrieval import Retriever, build_document_corpus
from memory.memory_store import MemoryStore
from utils.vector_store import VectorStore


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = DATA_DIR / "vector_index"

# Load environment variables from a local `.env` file.
# The app uses Gemini only if `GEMINI_API_KEY` is set.
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


st.set_page_config(page_title="Agentic AI Customer Support System", layout="wide")
st.title("Agentic AI Customer Support System")
st.caption("Real-time step streaming + admin observability dashboard (mock tools + JSON datasets).")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return json.loads(raw)


@st.cache_resource(show_spinner=False)
def build_system(force_rebuild: bool) -> LangGraphSupportActionAgent:
    knowledge_base = _load_json(DATA_DIR / "knowledge_base.json")
    tickets_seed = _load_json(DATA_DIR / "support_tickets.json")

    docs, metadatas = build_document_corpus(knowledge_base, tickets_seed)

    vector_store = VectorStore.load_or_create(
        index_dir=INDEX_DIR,
        documents=docs,
        metadatas=metadatas,
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        force_rebuild=force_rebuild,
    )

    retriever = Retriever(vector_store=vector_store)
    classifier = IntentClassifier(knowledge_base=knowledge_base)
    planner = Planner()

    memory_store = MemoryStore(data_dir=DATA_DIR)

    return LangGraphSupportActionAgent(
        classifier=classifier,
        retriever=retriever,
        planner=planner,
        memory_store=memory_store,
        vector_store=vector_store,
        vector_index_dir=str(INDEX_DIR),
    )


@st.cache_resource(show_spinner=False)
def load_users() -> list[dict[str, Any]]:
    return _load_json(DATA_DIR / "users.json")


def user_badge(final_decision: str, final_confidence: float) -> None:
    if final_decision == "auto_resolve":
        st.success(f"Decision: Auto-resolve | Confidence: {final_confidence:.2f}")
    elif final_decision == "ask_clarification":
        st.warning(f"Decision: Ask for clarification | Confidence: {final_confidence:.2f}")
    else:
        st.error(f"Decision: Escalate to admin | Confidence: {final_confidence:.2f}")


def render_admin_dashboard(memory_store: MemoryStore) -> None:
    tickets = memory_store.load_tickets()
    if not tickets:
        st.info("No tickets yet.")
        return

    df = pd.DataFrame(tickets)
    # Ensure columns exist for filtering.
    for col in ["ticket_id", "user_id", "category", "status", "created_at", "final_confidence", "final_decision", "intent_confidence"]:
        if col not in df.columns:
            df[col] = None

    st.subheader("Ticket Dashboard")

    statuses = sorted(df["status"].dropna().unique().tolist())
    categories = sorted(df["category"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", options=["All"] + statuses, index=0)
    with col2:
        category_filter = st.selectbox("Category", options=["All"] + categories, index=0)
    with col3:
        min_conf = st.slider("Min final confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    filtered = df.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]
    filtered = filtered[pd.to_numeric(filtered["final_confidence"], errors="coerce").fillna(0.0) >= min_conf]
    filtered = filtered.sort_values(by="created_at", ascending=False) if "created_at" in filtered.columns else filtered

    st.write(f"Showing {len(filtered)} tickets")
    st.dataframe(
        filtered[["ticket_id", "user_id", "category", "status", "final_decision", "final_confidence", "created_at"]].reset_index(drop=True),
        use_container_width=True,
    )

    ticket_ids = filtered["ticket_id"].dropna().astype(str).tolist()
    if not ticket_ids:
        return

    selected_id = st.selectbox("Select ticket to inspect", options=ticket_ids, index=0)
    ticket = next(t for t in tickets if str(t.get("ticket_id")) == str(selected_id))

    st.subheader(f"System Transparency: {ticket.get('ticket_id')}")
    st.markdown(f"**User query:** {ticket.get('query')}")
    st.markdown(f"**Classifier output:** {ticket.get('category')} (intent conf={ticket.get('intent_confidence')})")
    st.markdown(f"**RAG evidence:** retrieval_similarity_unit={ticket.get('retrieval_similarity_unit')}")
    st.markdown(f"**Final decision:** {ticket.get('final_decision')} | final_confidence={ticket.get('final_confidence')}")
    st.markdown(f"**Resolution:** {ticket.get('resolution')}")

    st.markdown("---")
    st.subheader("Execution Trace (step-by-step)")

    trace = ticket.get("agent_trace", []) or []
    for i, step in enumerate(trace):
        title = step.get("step", f"step_{i}")
        with st.expander(f"{i+1}. {title}", expanded=False):
            if step.get("streaming_msg"):
                st.write(step.get("streaming_msg"))
            st.write(f"confidence: {step.get('confidence')}")
            st.write(f"tool_name: {step.get('tool_name')}")
            st.write("input_data:")
            st.code(json.dumps(step.get("input_data", None), ensure_ascii=False, indent=2), language="json")
            st.write("output_data:")
            st.code(json.dumps(step.get("output_data", None), ensure_ascii=False, indent=2), language="json")


def render_user_view(agent: LangGraphSupportActionAgent, memory_store: MemoryStore, user_id: str) -> None:
    st.subheader("Support Chat")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.caption(f"Logged in as `{user_id}`")

        # Show message-level chat history for this user.
        history = memory_store.get_user_tickets(user_id)
        tickets_sorted = sorted(history, key=lambda x: str(x.get("created_at", "")))
        if tickets_sorted:
            for t in tickets_sorted[-8:]:
                for msg in t.get("messages", []) or []:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                    with st.chat_message(role):
                        st.write(content)

        st.divider()

        # Quick picker helps users provide the missing identifier (ORDxxxxx)
        # which unlocks tool calls for payment/delivery/order workflows.
        orders_all = _load_json(DATA_DIR / "orders.json")
        user_orders = [o for o in (orders_all or []) if str(o.get("user_id")) == str(user_id)]
        if user_orders:
            with st.expander("Quick pick from your mock orders", expanded=False):
                options = []
                for o in user_orders:
                    oid = o.get("order_id")
                    if not oid:
                        continue
                    label = f"{oid}|{o.get('order_status')}|{o.get('payment_status')}"
                    options.append(str(label))
                options = options[:10]
                chosen = st.selectbox("Select an order id (optional)", options=[""] + options)
                if chosen:
                    st.session_state["order_id"] = str(chosen).split("|")[0]

        order_id = st.text_input(
            "Order ID (optional, e.g., `ORD00012`)",
            value=st.session_state.get("order_id", ""),
            key="order_id",
            placeholder="Paste your order id here to get faster resolution",
        )

        user_query = st.chat_input("Describe your issue (mention payment/delivery/account + any order id you have)...")
        if not user_query:
            return

        composed_query = user_query
        if order_id and isinstance(order_id, str) and order_id.strip():
            composed_query = f"{user_query}\nOrder: {order_id.strip()}"

    with right:
        st.markdown("### Live Execution Steps")
        steps_box = st.empty()
        final_box = st.empty()
        badge_box = st.empty()

    steps: list[str] = []

    def stream_cb(evt: dict[str, Any]) -> None:
        if evt.get("type") == "step":
            msg = evt.get("message")
            if msg:
                steps.append(str(msg))
                steps_box.markdown("".join(["- " + s + "\n" for s in steps[-15:]]))

    # Stream step updates while backend runs.
    with left:
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            final_text_placeholder = st.empty()
            final_text_placeholder.write("Working through your request...")

            saved = agent.handle_ticket(user_id=user_id, query=composed_query, stream_callback=stream_cb)
            final_text_placeholder.empty()
            final_box.write(saved.get("resolution"))
            user_badge(saved.get("final_decision"), float(saved.get("final_confidence", 0.0)))


def main() -> None:
    force_rebuild = st.sidebar.checkbox("Rebuild vector index (slow, dev)", value=False)
    agent = build_system(force_rebuild=force_rebuild)
    memory_store = MemoryStore(data_dir=DATA_DIR)
    users = load_users()

    # Simple auth simulation: username-based access with shared password.
    ROLE_TO_USER_ID = {"User 1": "user_1", "User 2": "user_2", "User 3": "user_3"}
    ALIASES = {
        "user1": "user_1",
        "user_1": "user_1",
        "user2": "user_2",
        "user_2": "user_2",
        "user3": "user_3",
        "user_3": "user_3",
    }
    ADMIN_USERNAME = "admin"
    SHARED_PASSWORD = "1234"

    role = st.sidebar.radio("Select role", options=["User 1", "User 2", "User 3", "Admin"], index=0)

    # Reset auth when switching roles.
    if st.session_state.get("active_role") != role:
        st.session_state["active_role"] = role
        st.session_state.pop("auth_user_id", None)
        st.session_state.pop("auth_ok", None)

    if role == "Admin":
        st.sidebar.markdown("Admin mode: full tracing + confidence routing.")
        auth_ok = st.session_state.get("auth_ok") is True and st.session_state.get("auth_user_id") == ADMIN_USERNAME
        if not auth_ok:
            st.subheader("Admin Login")
            with st.form("admin_login_form"):
                username = st.text_input("Username", value=ADMIN_USERNAME)
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

            if submitted:
                if password == SHARED_PASSWORD and username.strip().lower() == ADMIN_USERNAME:
                    st.session_state["auth_user_id"] = ADMIN_USERNAME
                    st.session_state["auth_ok"] = True
                    st.success("Admin login successful.")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Use password `1234`.")
            return

        render_admin_dashboard(memory_store)
        return

    # User auth
    expected_user_id = ROLE_TO_USER_ID[role]
    auth_ok = st.session_state.get("auth_ok") is True and st.session_state.get("auth_user_id") == expected_user_id
    if not auth_ok:
        st.subheader("User Login")
        st.caption("Use username `user1`/`user2`/`user3` and password `1234`.")
        with st.form("user_login_form"):
            username = st.text_input("Username", value=expected_user_id.replace("_", ""))
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            normalized = ALIASES.get(username.strip().lower(), None)
            if password == SHARED_PASSWORD and normalized == expected_user_id:
                st.session_state["auth_user_id"] = expected_user_id
                st.session_state["auth_ok"] = True
                st.success(f"Login successful. Access granted to `{expected_user_id}`.")
                st.rerun()
            else:
                st.error(f"Invalid credentials for {role}. Try username `{expected_user_id.replace('_', '')}` and password `1234`.")
        return

    st.sidebar.success(f"Logged in as: {st.session_state['auth_user_id']}")
    render_user_view(agent=agent, memory_store=memory_store, user_id=st.session_state["auth_user_id"])


if __name__ == "__main__":
    main()


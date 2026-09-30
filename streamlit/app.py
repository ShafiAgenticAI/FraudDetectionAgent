import uuid
from datetime import datetime
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Compliance Copilot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.05);
        text-align: left;
        font-weight: 500;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(56,189,248,0.18);
        border-color: rgba(56,189,248,0.45);
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #0ea5e9, #0284c7);
        border: none;
        color: #ffffff;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.7rem !important;
    }

    /* ---- Main content ---- */
    [data-testid="stAppViewContainer"] .main {
        background: #f8fafc;
    }
    h1, h2, h3 {
        color: #0f172a;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border-color: #e2e8f0 !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #0ea5e9, #0284c7);
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton button[kind="secondary"] {
        border-radius: 10px;
    }

    /* ---- Chat ---- */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
        background: #ffffff;
        border: 1px solid #eef2f7;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = ""

if "view" not in st.session_state:
    st.session_state["view"] = "chat"

if "chat_sessions" not in st.session_state:
    # session_id doubles as the backend's chat-session cache key (see
    # app/services/session_store.py), so a chat you switch to in the UI
    # is the same conversation the backend remembers the last 10
    # exchanges of.
    first_id = str(uuid.uuid4())
    st.session_state["chat_sessions"] = {
        first_id: {"title": "New chat", "messages": [], "created": datetime.now()}
    }
    st.session_state["chat_order"] = [first_id]
    st.session_state["active_chat_id"] = first_id


def _new_chat() -> str:
    new_id = str(uuid.uuid4())
    st.session_state["chat_sessions"][new_id] = {
        "title": "New chat",
        "messages": [],
        "created": datetime.now(),
    }
    st.session_state["chat_order"].insert(0, new_id)
    st.session_state["active_chat_id"] = new_id
    return new_id


def _delete_chat(chat_id: str) -> None:
    try:
        requests.delete(f"{API_BASE_URL}/api/chat/session/{chat_id}", timeout=15)
    except requests.RequestException:
        pass  # local cleanup still proceeds

    st.session_state["chat_sessions"].pop(chat_id, None)
    st.session_state["chat_order"] = [
        c for c in st.session_state["chat_order"] if c != chat_id
    ]

    if st.session_state["active_chat_id"] == chat_id:
        if st.session_state["chat_order"]:
            st.session_state["active_chat_id"] = st.session_state["chat_order"][0]
        else:
            _new_chat()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.25rem 0 1rem 0;">
            <h2 style="margin:0; color:#f8fafc;">📘 Compliance Copilot</h2>
            <p style="margin:0.15rem 0 0 0; color:#94a3b8; font-size:0.85rem;">
                Regulatory document AI assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    nav_items = [
        ("chat", "💬  Compliance Chat"),
        ("analysis", "📊  Document Analysis"),
        ("documents", "📄  Documents"),
        ("analytics", "📈  Analytics"),
    ]

    for key, label in nav_items:
        active = st.session_state["view"] == key
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state["view"] = key
            st.rerun()

    if st.session_state["view"] == "chat":
        st.divider()

        if st.button("➕  New chat", use_container_width=True):
            _new_chat()
            st.rerun()

        st.caption("Recent chats")

        for chat_id in list(st.session_state["chat_order"]):
            session = st.session_state["chat_sessions"].get(chat_id)
            if not session:
                continue

            is_active = chat_id == st.session_state["active_chat_id"]
            title = session["title"] or "New chat"
            label = title if len(title) <= 28 else title[:27] + "…"

            col_title, col_delete = st.columns([5, 1])

            with col_title:
                if st.button(
                    label,
                    key=f"switch_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_chat_id"] = chat_id
                    st.rerun()

            with col_delete:
                if st.button("🗑", key=f"del_{chat_id}", use_container_width=True):
                    _delete_chat(chat_id)
                    st.rerun()

# --------------------------------------------------------------------------
# Documents view
# --------------------------------------------------------------------------


def render_documents_view() -> None:
    st.title("📄 Documents")
    st.caption("Upload a regulatory PDF, index it for search, or remove it entirely.")

    with st.container(border=True):
        st.subheader("Upload")
        uploaded = st.file_uploader(
            "Upload a regulatory PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded and st.button("Upload PDF", type="primary"):
            try:
                with st.spinner("Uploading..."):
                    response = requests.post(
                        f"{API_BASE_URL}/api/documents/upload",
                        files={
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=120,
                    )

                if response.ok:
                    data = response.json()
                    st.session_state["last_filename"] = data["filename"]
                    st.success("PDF uploaded successfully.")
                    st.rerun()
                else:
                    st.error(response.text)

            except requests.RequestException as exc:
                st.error(f"Backend connection error: {exc}")

    registry = []

    with st.container(border=True):
        st.subheader("Indexed documents")

        try:
            response = requests.get(f"{API_BASE_URL}/api/documents/", timeout=30)

            if response.ok:
                registry = response.json().get("documents", [])

                if registry:
                    st.dataframe(
                        [
                            {
                                "Filename": doc["filename"],
                                "Status": doc["status"],
                                "Chunks": doc["chunks_count"],
                                "Source": doc["source"],
                                "Uploaded": doc["upload_date"][:19].replace("T", " "),
                                "Indexed": (
                                    (doc["indexed_at"] or "")[:19].replace("T", " ")
                                    if doc.get("indexed_at")
                                    else "—"
                                ),
                            }
                            for doc in registry
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No documents uploaded yet.")
            else:
                st.error(response.text)

        except requests.RequestException as exc:
            st.error(f"Backend connection error: {exc}")

    filenames = [doc["filename"] for doc in registry]

    if filenames:
        default_index = (
            filenames.index(st.session_state["last_filename"])
            if st.session_state["last_filename"] in filenames
            else 0
        )

        with st.container(border=True):
            st.subheader("Manage document")

            filename = st.selectbox(
                "Select a document", filenames, index=default_index
            )
            st.session_state["last_filename"] = filename

            force_reindex = st.checkbox(
                "Force re-index (skip the unchanged-content check)",
                value=False,
                help=(
                    "By default, re-indexing a file whose content hasn't "
                    "changed since the last ingest is skipped automatically."
                ),
            )

            col_index, col_delete = st.columns(2)

            with col_index:
                if st.button(
                    "⚡ Index / Create Embeddings",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner(
                            "Extracting pages, chunking and creating embeddings..."
                        ):
                            response = requests.post(
                                f"{API_BASE_URL}/api/documents/ingest",
                                json={
                                    "filename": filename,
                                    "force": force_reindex,
                                },
                                timeout=900,
                            )

                        if response.ok:
                            st.success("Document indexed successfully.")
                            st.json(response.json())
                            st.rerun()
                        else:
                            st.error(response.text)

                    except requests.RequestException as exc:
                        st.error(f"Backend connection error: {exc}")

            with col_delete:
                if st.button(
                    "🗑️ Delete document & index",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner(
                            "Removing document and its indexed chunks..."
                        ):
                            response = requests.delete(
                                f"{API_BASE_URL}/api/documents/{filename}",
                                timeout=60,
                            )

                        if response.ok:
                            st.success(
                                "Document and its indexed chunks were removed."
                            )
                            st.session_state["last_filename"] = ""
                            st.rerun()
                        else:
                            st.error(response.text)

                    except requests.RequestException as exc:
                        st.error(f"Backend connection error: {exc}")


# --------------------------------------------------------------------------
# Compliance Chat view
# --------------------------------------------------------------------------


def render_chat_view() -> None:
    active_id = st.session_state["active_chat_id"]
    active = st.session_state["chat_sessions"][active_id]

    st.title(active["title"] if active["messages"] else "💬 Compliance Chat")
    st.caption(
        "Grounded answers with citations, from the regulatory documents "
        "you've indexed. Follow-ups understand the last 10 exchanges."
    )

    if not active["messages"]:
        st.markdown(
            """
            <div style="text-align:center; padding: 3.5rem 1rem; color:#64748b;">
                <div style="font-size:2.5rem;">💬</div>
                <h3 style="color:#334155; margin-bottom:0.25rem;">
                    Ask about your regulatory documents
                </h3>
                <p style="margin:0;">
                    Try: "What are the recordkeeping requirements?" or
                    "What's the deadline for filing a disclosure?"
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in active["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                if message.get("intent") == "domain_query" and not message.get(
                    "grounded", True
                ):
                    st.warning(
                        "⚠️ This answer had low overlap with the retrieved "
                        "source text -- treat it as unverified and "
                        "double-check the sources below."
                    )

                citations = message.get("citations") or []

                if citations:
                    with st.expander(f"📚 Sources ({len(citations)})"):
                        for index, citation in enumerate(citations, start=1):
                            st.markdown(
                                f"**{index}. {citation['document']}**  \n"
                                f"Page: `{citation['page']}`  \n"
                                f"Chunk: `{citation['chunk_id']}`"
                            )

    question = st.chat_input(
        "Ask a compliance question... (e.g. 'does it apply to contractors?')"
    )

    if question:
        active["messages"].append({"role": "user", "content": question})

        if active["title"] == "New chat" and len(active["messages"]) == 1:
            active["title"] = (
                question if len(question) <= 40 else question[:39] + "…"
            )

        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_BASE_URL}/api/chat/query",
                    json={
                        "question": question,
                        "top_k": 5,
                        "session_id": active_id,
                    },
                    timeout=120,
                )

            if response.ok:
                data = response.json()
                active["messages"].append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "intent": data.get("intent"),
                        "grounded": data.get("grounded", True),
                        "citations": data.get("citations", []),
                    }
                )
            else:
                active["messages"].append(
                    {"role": "assistant", "content": f"⚠️ Error: {response.text}"}
                )

        except requests.RequestException as exc:
            active["messages"].append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Backend connection error: {exc}",
                }
            )

        st.session_state["chat_order"].remove(active_id)
        st.session_state["chat_order"].insert(0, active_id)

        st.rerun()


# --------------------------------------------------------------------------
# Document Analysis view
# --------------------------------------------------------------------------


def render_analysis_view() -> None:
    st.title("📊 Document Analysis")
    st.caption(
        "Generate an executive summary or extract structured compliance "
        "obligations from an indexed document."
    )

    filename = st.text_input(
        "Document filename",
        value=st.session_state["last_filename"],
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Executive summary")

            if st.button(
                "Generate summary", type="primary", use_container_width=True
            ):
                if not filename.strip():
                    st.warning("Enter a document filename.")
                else:
                    try:
                        with st.spinner("Generating executive summary..."):
                            response = requests.post(
                                f"{API_BASE_URL}/api/analysis/summary",
                                json={"filename": filename},
                                timeout=900,
                            )

                        if response.ok:
                            st.write(response.json()["summary"])
                        else:
                            st.error(response.text)

                    except requests.RequestException as exc:
                        st.error(f"Backend connection error: {exc}")

    with col2:
        with st.container(border=True):
            st.subheader("Compliance obligations")

            if st.button(
                "Extract obligations", type="primary", use_container_width=True
            ):
                if not filename.strip():
                    st.warning("Enter a document filename.")
                else:
                    try:
                        with st.spinner("Extracting compliance obligations..."):
                            response = requests.post(
                                f"{API_BASE_URL}/api/analysis/obligations",
                                json={"filename": filename},
                                timeout=900,
                            )

                        if response.ok:
                            data = response.json()
                            st.caption(f"{data['count']} obligation(s) found")

                            if data["obligations"]:
                                st.dataframe(
                                    data["obligations"],
                                    use_container_width=True,
                                )
                            else:
                                st.info(
                                    "No explicit obligations were extracted."
                                )
                        else:
                            st.error(response.text)

                    except requests.RequestException as exc:
                        st.error(f"Backend connection error: {exc}")


# --------------------------------------------------------------------------
# Analytics view
# --------------------------------------------------------------------------


def render_analytics_view() -> None:
    st.title("📈 Analytics")
    st.caption(
        "Live numbers behind the MVP Success Criteria, computed from the "
        "query audit log and analysis-run timings."
    )

    try:
        response = requests.get(f"{API_BASE_URL}/api/analytics/kpis", timeout=30)
    except requests.RequestException as exc:
        st.error(f"Backend connection error: {exc}")
        return

    if not response.ok:
        st.error(response.text)
        return

    kpis = response.json()
    targets = kpis.get("targets", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.metric(
                "Avg response time",
                f"{kpis['avg_response_time_ms'] / 1000:.2f} s",
                help=f"Target: < {targets.get('response_time_ms', 5000) / 1000:.0f} s",
            )
            st.caption(f"{kpis['total_queries']} total queries logged")

    with col2:
        with st.container(border=True):
            st.metric(
                "Citation coverage",
                f"{kpis['citation_coverage_pct']}%",
                help=f"Target: {targets.get('citation_coverage_pct', 100)}%",
            )
            st.caption("Domain queries answered with at least one citation")

    with col3:
        with st.container(border=True):
            st.metric(
                "Grounded answer rate",
                f"{kpis['grounded_rate_pct']}%",
                help="Answers passing the groundedness heuristic",
            )
            st.caption("See rag_service._is_grounded")

    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):
            st.metric(
                "Avg summary time",
                f"{kpis['avg_summary_time_ms'] / 1000:.2f} s",
                help=f"Target: < {targets.get('summary_time_ms', 15000) / 1000:.0f} s",
            )

    with col5:
        with st.container(border=True):
            st.metric(
                "Avg obligation extraction time",
                f"{kpis['avg_obligation_time_ms'] / 1000:.2f} s",
            )

    with col6:
        with st.container(border=True):
            st.metric(
                "Indexed documents",
                f"{kpis['indexed_documents']} / {kpis['total_documents']}",
            )

    st.divider()
    with st.container(border=True):
        st.subheader("🧪 Ragas RAG Evaluation")
        st.caption("Offline evaluation of the 40-question golden set. It does not run during normal chat.")
        if st.button("▶️ Run Ragas Evaluation", key="run_ragas_eval"):
            with st.spinner("Running golden-set evaluation with Ragas. This may take several minutes..."):
                try:
                    rr = requests.post(f"{API_BASE_URL}/api/analytics/ragas/run", timeout=1900)
                    if rr.ok:
                        data = rr.json()
                        st.success(f"Evaluation complete — {data.get('golden_questions', 0)} golden questions evaluated.")
                        metrics = data.get("metrics", {})
                        cols = st.columns(max(1, min(5, len(metrics) + 1)))
                        for idx, (name, value) in enumerate(metrics.items()):
                            cols[idx % len(cols)].metric(name.replace("_", " ").title(), f"{float(value):.3f}")
                        st.metric("Citation page hit rate", f"{float(data.get('citation_page_hit_rate', 0))*100:.1f}%")
                    else:
                        st.error(rr.text)
                except requests.RequestException as exc:
                    st.error(f"Backend connection error: {exc}")

        try:
            rh = requests.get(f"{API_BASE_URL}/api/analytics/ragas", params={"limit": 10}, timeout=30)
            if rh.ok:
                evaluations = rh.json().get("evaluations", [])
                if evaluations:
                    st.dataframe(
                        [{"Run": e["run_timestamp"], "Questions": e["golden_questions"], **{k: round(float(v), 3) for k, v in e["metrics"].items()}, "Citation page hit rate": round(float(e["citation_page_hit_rate"]), 3)} for e in evaluations],
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No Ragas evaluation runs yet.")
        except requests.RequestException:
            pass

    with st.container(border=True):
        st.subheader("Recent queries")

        try:
            log_response = requests.get(
                f"{API_BASE_URL}/api/analytics/queries",
                params={"limit": 25},
                timeout=30,
            )

            if log_response.ok:
                rows = log_response.json().get("queries", [])

                if rows:
                    st.dataframe(
                        [
                            {
                                "Time": row["timestamp"][:19].replace("T", " "),
                                "Question": row["question"],
                                "Intent": row["intent"],
                                "Grounded": bool(row["grounded"]),
                                "Citations": row["citation_count"],
                                "Response (ms)": row["response_time_ms"],
                            }
                            for row in rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No queries logged yet.")
            else:
                st.error(log_response.text)

        except requests.RequestException as exc:
            st.error(f"Backend connection error: {exc}")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

if st.session_state["view"] == "documents":
    render_documents_view()
elif st.session_state["view"] == "analysis":
    render_analysis_view()
elif st.session_state["view"] == "analytics":
    render_analytics_view()
else:
    render_chat_view()

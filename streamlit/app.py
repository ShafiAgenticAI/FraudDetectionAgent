import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Regulatory Compliance Copilot",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Regulatory Compliance Copilot")
st.caption(
    "Local MVP — regulatory document ingestion, grounded Q&A, "
    "citations, summaries, and compliance obligations."
)

if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = ""

tab_docs, tab_chat, tab_analysis = st.tabs(
    [
        "📄 Documents",
        "💬 Compliance Chat",
        "📊 Document Analysis",
    ]
)


with tab_docs:
    st.header("Upload & Index Regulatory PDF")

    uploaded = st.file_uploader(
        "Upload a regulatory PDF",
        type=["pdf"],
    )

    if uploaded and st.button("Upload PDF"):
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
            else:
                st.error(response.text)

        except requests.RequestException as exc:
            st.error(f"Backend connection error: {exc}")

    filename = st.session_state["last_filename"]

    if filename:
        st.info(f"Selected document: {filename}")

        if st.button("Index / Create Embeddings"):
            try:
                with st.spinner(
                    "Extracting pages, chunking and creating embeddings..."
                ):
                    response = requests.post(
                        f"{API_BASE_URL}/api/documents/ingest",
                        json={"filename": filename},
                        timeout=900,
                    )

                if response.ok:
                    st.success("Document indexed successfully.")
                    st.json(response.json())
                else:
                    st.error(response.text)

            except requests.RequestException as exc:
                st.error(f"Backend connection error: {exc}")


with tab_chat:
    st.header("Ask the Regulations")

    question = st.text_area(
        "Ask a compliance question",
        placeholder=(
            "Example: What are the recordkeeping requirements "
            "under this regulation?"
        ),
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Searching indexed regulatory content..."):
                    response = requests.post(
                        f"{API_BASE_URL}/api/chat/query",
                        json={
                            "question": question,
                            "top_k": 5,
                        },
                        timeout=120,
                    )

                if response.ok:
                    data = response.json()

                    st.subheader("Answer")
                    st.write(data["answer"])

                    st.subheader("Sources")

                    for index, citation in enumerate(
                        data.get("citations", []),
                        start=1,
                    ):
                        st.markdown(
                            f"**{index}. {citation['document']}**  \n"
                            f"Page: `{citation['page']}`  \n"
                            f"Chunk: `{citation['chunk_id']}`"
                        )
                else:
                    st.error(response.text)

            except requests.RequestException as exc:
                st.error(f"Backend connection error: {exc}")


with tab_analysis:
    st.header("Document Analysis")

    filename = st.text_input(
        "Document filename",
        value=st.session_state["last_filename"],
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Executive Summary"):
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
                        st.subheader("Executive Summary")
                        st.write(response.json()["summary"])
                    else:
                        st.error(response.text)

                except requests.RequestException as exc:
                    st.error(f"Backend connection error: {exc}")

    with col2:
        if st.button("Extract Compliance Obligations"):
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

                        st.subheader(
                            f"Obligations ({data['count']})"
                        )

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

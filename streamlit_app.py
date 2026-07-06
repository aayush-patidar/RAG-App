import asyncio
import os
import time
import uuid
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📄",
    layout="centered",
)


# Creates one unique user ID per browser session.
# In production, replace this with the logged-in user's database ID.
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_appa",
        is_production=False,
    )


def save_uploaded_pdf(file, user_id: str) -> Path:
    uploads_dir = Path("uploads") / user_id
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())

    return file_path


async def send_rag_ingest_event(pdf_path: Path, user_id: str) -> None:
    client = get_inngest_client()

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
                "user_id": user_id,
            },
        )
    )


async def send_rag_query_event(
    question: str,
    top_k: int,
    user_id: str,
):
    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "user_id": user_id,
            },
        )
    )

    return result[0]


def inngest_api_base() -> str:
    return os.getenv(
        "INNGEST_API_BASE",
        "http://127.0.0.1:8288/v1",
    )


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{inngest_api_base()}/events/{event_id}/runs"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return data.get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5,
) -> dict:
    start_time = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)

        if runs:
            run = runs[0]
            status = run.get("status")

            if status:
                last_status = status

            if status in (
                "Completed",
                "Succeeded",
                "Success",
                "Finished",
            ):
                return run.get("output") or {}

            if status in (
                "Failed",
                "Cancelled",
            ):
                raise RuntimeError(
                    f"Inngest function failed. Status: {status}"
                )

        if time.time() - start_time > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for response. Last status: {last_status}"
            )

        time.sleep(poll_interval_s)


st.title("PDF RAG Assistant")

st.caption(
    f"Current session ID: {st.session_state.user_id[:8]}"
)

st.divider()

st.subheader("Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    if st.button("Ingest PDF"):
        try:
            with st.spinner("Uploading and processing PDF..."):
                pdf_path = save_uploaded_pdf(
                    uploaded_file,
                    st.session_state.user_id,
                )

                asyncio.run(
                    send_rag_ingest_event(
                        pdf_path,
                        st.session_state.user_id,
                    )
                )

                time.sleep(0.3)

            st.success(
                f"PDF ingestion started: {uploaded_file.name}"
            )

        except Exception as error:
            st.error(f"Ingestion error: {error}")


st.divider()

st.subheader("Ask a question about your PDF")

with st.form("rag_query_form"):
    question = st.text_input(
        "Your question",
        placeholder="Example: What skills are mentioned?",
    )

    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button("Ask question")


if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching your PDF..."):
                event_id = asyncio.run(
                    send_rag_query_event(
                        question.strip(),
                        int(top_k),
                        st.session_state.user_id,
                    )
                )

                output = wait_for_run_output(event_id)

            answer = output.get("answer", "")
            sources = output.get("sources", [])

            st.subheader("Answer")
            st.write(answer or "No answer returned.")

            if sources:
                st.subheader("Sources")

                for source in sources:
                    st.write(f"- {source}")

        except Exception as error:
            st.error(f"Query error: {error}")
import streamlit as st
import langchain_helper as lch
import textwrap
import os
import tempfile

st.title("📄 Document Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        uploaded_file = st.sidebar.file_uploader(
            label="Upload a PDF document",
            type=["pdf"]
        )

        query = st.sidebar.text_area(
            label="Ask a question about the document",
            max_chars=200,
            key="query"
        )

        submit_button = st.form_submit_button(label="Submit")


if submit_button:
    if not uploaded_file or not query:
        st.info("Please upload a PDF and enter a question.")
        st.stop()

  # Save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    db = lch.create_db_from_documents(pdf_path)
    response, docs = lch.get_response_from_query(db, query)

    st.subheader("Answer")
    st.text(textwrap.fill(response, width=85))

    # Optional: show source chunks
    with st.expander("Show source document chunks"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1} (page {doc.metadata.get('page')})**")
            st.write(doc.page_content[:500])

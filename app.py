import streamlit as st
import tempfile
from rag_chain import load_docs, create_vector_store, get_rag_chain

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot with LangChain + Groq")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    st.success("✅ File uploaded successfully!")

    with st.spinner("🔄 Processing the document..."):
        documents = load_docs(temp_file_path)
        vector_store = create_vector_store(documents)
        rag_chain = get_rag_chain(vector_store)

    st.success("📄 Document processed. Ask your questions below:")

    user_question = st.text_input("💬 Ask your question here:")

    if user_question:
        with st.spinner("🤖 Generating answer..."):
            response = rag_chain({"query": user_question})
            answer = response["result"]
            sources = response["source_documents"]

        st.subheader("🧠 Answer:")
        st.write(answer)

        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content[:500] + "...")

else:
    st.info("📤 Please upload a PDF file to start chatting.")

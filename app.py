# import streamlit as st
# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# from utils.rag_pipeline  import create_qa_chain

# from dotenv import load_dotenv

# load_dotenv()

# st.set_page_config(page_title="AI PDF Chatbot")

# st.title(" AI PDF Chatbot (RAG)")

# st.write("App started...")  # 👈 DEBUG LINE

# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# if uploaded_file is not None:
#     st.write("File detected")  # 👈 DEBUG

#     with open("temp.pdf", "wb") as f:
#         f.write(uploaded_file.read())

#     st.success("PDF uploaded successfully!")

#     try:
#         st.write("Creating QA chain...")  # 👈 DEBUG
#         qa_chain = create_qa_chain("temp.pdf")

#         query = st.text_input("Ask a question from the PDF")

#         if query:
#             with st.spinner("Thinking..."):
#                 answer = qa_chain.run(query)
#                 st.write("### Answer:")
#                 st.write(answer)

#     except Exception as e:
#         st.error(f"Error: {e}")

import streamlit as st
from utils.rag_pipeline import create_qa_chain
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("GROQ_API_KEY"))

st.title("PDF Chatbot ")

pdf_path =r"C:\Users\gayat\Downloads\Rakesh_HPC-cv.pdf"

qa_chain = create_qa_chain(pdf_path)

query = st.text_input("Ask something from PDF:")

if query:
    result = qa_chain.run(query)
    st.write(result)
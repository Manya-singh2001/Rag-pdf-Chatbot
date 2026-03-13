import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64

import os

# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# updated
from langchain_huggingface import HuggingFaceEmbeddings




from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

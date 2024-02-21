import os
import sys
import streamlit as st

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "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"

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

if PERSIST and os.path.exists("persist"):
    st.write("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # loader = TextLoader("data/data.txt") # Use this line if you only need a specific file e.g. data.txt
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

st.title("Conversational AI with Langchain and OpenAI")
query = st.text_input("Enter a prompt:")

if st.button("Ask"):
    if query in ['quit', 'q', 'exit']:
        st.write("Exiting the chat.")
        st.stop()
    result = chain({"question": query, "chat_history": chat_history})
    st.write(result['answer'])
    chat_history.append((query, result['answer']))

st.write("Chat History:")
for i, (input_query, answer) in enumerate(chat_history):
    st.write(f"{i + 1}. User: {input_query}")
    st.write(f"{i + 1}. AI: {answer}")
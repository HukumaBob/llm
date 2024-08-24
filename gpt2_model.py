from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

embeddings = OllamaEmbeddings()

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")


output_parser = StrOutputParser()
llm = Ollama(model="llama2")
# llm.invoke("how can langsmith help with testing?")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])
chain = prompt | llm | output_parser
chain.invoke({"input": "how can langsmith help with testing?"})
document_chain = create_stuff_documents_chain(llm, prompt)
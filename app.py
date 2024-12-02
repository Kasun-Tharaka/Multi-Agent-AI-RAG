import streamlit as st
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_groq import ChatGroq
import os
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)

# docs_list
# doc_splits

ASTRA_DB_APPLICATION_TOKEN="AstraCS:TkLSGODCCZtZfQNzwtCbpnTl:fb98a78d6b9a4ee069f0797ac6ad2383830863b239cd9cd08c30183fceca9c17"
ASTRA_DB_ID="71450f9f-a69c-4387-bfd9-aef0874d99ca"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

# update this code into access the model in local to do embeddings
# Or download model and call from here
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # need a token access

astra_vector_store = Cassandra(embedding=embeddings, table_name="MultiAIagentText", session=None, keyspace=None)

astra_vector_store.add_documents(doc_splits) # insertin into DB
print("Inserted %i Headline" % len(doc_splits))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store) 
retriever = astra_vector_store.as_retriever()
retriever.invoke("what is RAG")



api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki.run('What is Sri Sigiriya')
arxiv.run("What is SSD do")





# RouteQuery Class
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ..., description="Given a user question choose to route it to wikipedia or a vectorstore."
    )

groq_api_key = "pass"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.1-70b-Versatile")

structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt for routing
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# Routing Function
def route_question(state):
    question = state["question"]
    llm_response = route_prompt | structured_llm_router
    result = llm_response.invoke({"question": question})
    return result.datasource



class GraphState(TypedDict):

    question: str             # question
    generation: str           # LLM generation
    documents: List[str]      # list of documents



def retrieve(state):
    
    print("---RETRIEVE FROM VECTOR STORE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents[0].dict()['metadata']['description'], "question": question}



def wiki_search(state):
    
    print("---WIKIPIDIA SEARCH---")
    question = state["question"]
    print(question)

    docs = wiki.invoke({"query": question})
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    if hasattr(wiki_results, "page_content") and "Summary:" in wiki_results.page_content:
            summary_start = wiki_results.page_content.find("Summary:") + len("Summary:")
            summary = wiki_results.page_content[summary_start:].strip()

    return {"documents": summary, "question": question}



workflow = StateGraph(GraphState)

workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)

app = workflow.compile()



# Streamlit UI
st.title("RAG + Wikipedia Search Application")
st.markdown(
    """
    Enter a question, and the application will decide whether to search the RAG system or Wikipedia.
    """
)

# Input question
user_question = st.text_input("Ask a question:", "")

if st.button("Submit"):
    if user_question.strip():
        st.write(f"**Question:** {user_question}")
        
        for output in app.stream(user_question):
            for key, value in output.items():
                print(f"Node '{key}':")
        
    else:
        st.warning("Please enter a valid question.")

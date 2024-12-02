import cassio
ASTRA_DB_APPLICATION_TOKEN="AstraCS:TkLSGODCCZtZfQNzwtCbpnTl:fb98a78d6b9a4ee069f0797ac6ad2383830863b239cd9cd08c30183fceca9c17"
ASTRA_DB_ID="71450f9f-a69c-4387-bfd9-aef0874d99ca"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)

docs_list

doc_splits

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain.vectorstores.cassandra import Cassandra
astra_vector_store = Cassandra(embedding=embeddings, table_name="MultiAIagentText", session=None, keyspace=None)

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
astra_vector_store.add_documents(doc_splits) # insertin into DB
print("Inserted %i Headline" % len(doc_splits))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store) 
retriever = astra_vector_store.as_retriever()
retriever.invoke("what is RAG")

!pip install langchain_community
!pip install arxiv wikipedia

### Working With Tools
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki.run('What is Sri Sigiriya')

arxiv.run("What is SSD do")

### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


class RouteQuery(BaseModel):

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )


from google.colab import userdata 

from langchain_groq import ChatGroq
import os

groq_api_key = userdata.get('groqapiMultiAIagentRAG')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.1-70b-Versatile")

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

print(
    question_router.invoke(
        {"question": "who is Sri Lanka?"}
    )
)


print(
    question_router.invoke(
        {"question": "What are the types of agent memory?"}
        )
    )

def route_question(state):

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"

    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"



from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):

    question: str             # question
    generation: str           # LLM generation
    documents: List[str]      # list of documents

from langchain.schema import Document


def retrieve(state):

    print("---RETRIEVE FROM VECTOR STORE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):

    print("---WIKIPIDIA SEARCH---")
    question = state["question"]
    print(question)

    docs = wiki.invoke({"query": question})
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}


from langgraph.graph import END, StateGraph, START

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

from pprint import pprint

inputs = {
    "question": "What is agent?"
}

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")

    pprint("\n---\n")

pprint(value['documents'][0].dict()['metadata']['description'])

from pprint import pprint






inputs = {
    "question": "What is Sigiriya?"
}

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")

        document = value["documents"] 
        if hasattr(document, "page_content") and "Summary:" in document.page_content:
            summary_start = document.page_content.find("Summary:") + len("Summary:")
            summary = document.page_content[summary_start:].strip()
            pprint(f"Summary: {summary}")

    pprint("\n---\n")
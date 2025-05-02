from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load Mistral model
model = AutoModelForCausalLM.from_pretrained("./mistral-finetuned", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=gen)

# Retrieval Tool
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_store", embedding)

retriever_tool = Tool(
    name="Vector Search",
    func=lambda q: vectorstore.similarity_search(q, k=2),
    description="Useful for answering questions about documents"
)

# Math/Code Tool
python_tool = PythonREPLTool()

agent_executor = initialize_agent(
    tools=[retriever_tool, python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

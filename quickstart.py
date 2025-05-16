from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# For uniGPT we need to set the llm and embed_model

llm = OpenAILike(
    model       = "Llama-3.3-70B",
    api_base    = "https://gpt.uni-muenster.de/v1",
    api_key     = "sk-yd4l1I6RqeYQ8K-2WjtGQA",
)

embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2") # 384 Dimensions, 90 MB(!)

# https://docs.llamaindex.ai/en/stable/#30-second-quickstart

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("Was ist die Hauptstadt von Frankreich?")
print(response)

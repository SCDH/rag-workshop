from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = OpenAILike(
    model="Llama-3.3-70B",
    api_base="https://gpt.uni-muenster.de/v1",
    api_key="", # <-- Insert the UniGPT key here
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=False,
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./cache"  # Optional: cache embeddings locally
)

# Configure LlamaIndex settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# Load and index documents
print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

print("Building vector index...")
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

print("Interactive mode - Enter your queries (type 'quit' to exit):")

while True:
    user_query = input("\nQuery: ")
    if user_query.lower() == 'quit':
        break

    try:
        response = query_engine.query(user_query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
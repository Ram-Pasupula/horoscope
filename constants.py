import os
# from dotenv import load_dotenv
#from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, UnstructuredXMLLoader
# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

EMBED_CACHE_FOLDER = f"{ROOT_DIRECTORY}/../ca-va/embed"
MODELS_PATH = f"{ROOT_DIRECTORY}/../ca-va/models"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8


# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 4000
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

# If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

N_GPU_LAYERS = 0  # Llama-2-70B has 83 layers
N_BATCH = 512

# From experimenting with the Llama-2-7B-Chat-GGML model on 8GB VRAM, these values work:
# N_GPU_LAYERS = 20
# N_BATCH = 512


DOCUMENT_MAP = {
     ".xml":  UnstructuredXMLLoader,

    ".csv": CSVLoader
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-base"
#EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
#EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"
#EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"

#MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
#MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"
MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
#MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf" 
MODEL_BASENAME = "llama-2-7b-chat.Q5_K_M.gguf" 
#large, very low quality loss - recommended

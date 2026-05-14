from llama_index.embeddings.ollama import OllamaEmbedding
from configs.config import EMBEDDING_MODEL


def get_embedding_model():
    return OllamaEmbedding(model_name=EMBEDDING_MODEL)
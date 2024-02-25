from typing import Optional

from embedchain.config import BaseEmbedderConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.models import VectorDimensions

from embedchain.config.embedder.ollama import OllamaEmbedderConfig


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, config: Optional[OllamaEmbedderConfig] = None):
        super().__init__(config=config)

        from langchain_community.embeddings.ollama import OllamaEmbeddings as LangchainOllamaEmbeddings

        embeddings = LangchainOllamaEmbeddings(model=self.config.model, base_url=self.config.base_url)

        embedding_fn = BaseEmbedder._langchain_default_concept(embeddings)

        self.set_embedding_fn(embedding_fn=embedding_fn)

        vector_dimension = 768#self.config.vector_dimension

        self.set_vector_dimension(vector_dimension=vector_dimension)

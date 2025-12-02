"""
Text embedding provider using Hugging Face Transformers
"""

from typing import List, Optional
import numpy as np


class EmbeddingProvider:
    """Text embedding provider wrapper"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("sentence-transformers not installed, falling back to transformers")
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
    
    def batch_encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode a batch of texts to embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress_bar: Show progress bar
        
        Returns:
            embeddings: (N, embedding_dim) numpy array
        """
        if isinstance(self.model, object) and hasattr(self.model, 'encode'):
            # sentence-transformers or embedding models
            # Check if it's an embedding gemma model (has encode_query/encode_document)
            if hasattr(self.model, 'encode_query'):
                # Embedding Gemma style
                embeddings = self.model.encode_document(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True
                )
            else:
                # Standard sentence-transformers
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True
                )
            return embeddings.astype(np.float32)
        else:
            # transformers fallback
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding
                    embeddings_list.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            
            embeddings = np.vstack(embeddings_list)
            
            if normalize:
                embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
            return embeddings.astype(np.float32)


def get_embedding_provider(model_name: str) -> EmbeddingProvider:
    """
    Get an embedding provider instance
    
    Args:
        model_name: Model name (e.g., "google/embedding-gecko-text-3")
    
    Returns:
        EmbeddingProvider instance
    """
    return EmbeddingProvider(model_name)

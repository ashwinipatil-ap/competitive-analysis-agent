import os
import logging
import pandas as pd
from typing import List, Dict, Any

# LlamaIndex imports (organized so the file is importable even if dependencies are missing)
try:
    from llama_index.core import Document, VectorStoreIndex, ServiceContext, StorageContext
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.embeddings.cohere import CohereEmbedding
    from llama_index.core.indices.postprocessor import SentenceTransformerRerank
    LLAMA_OK = True
except Exception:
    LLAMA_OK = False

class RagPipeline:
    """Sets up LlamaIndex + Cohere embeddings over the competitors dataset."""

    def __init__(self, csv_path: str, cohere_api_key: str | None = None, top_k: int = 4):
        self.csv_path = csv_path
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.top_k = top_k
        self.index = None
        self._fallback_rows: list[dict] = []
        self._setup()

    def _setup(self) -> None:
        # Load data
        df = pd.read_csv(self.csv_path).fillna("")
        self._fallback_rows = df.to_dict(orient="records")

        if not LLAMA_OK or not self.cohere_api_key:
            logging.warning("RagPipeline running in FALLBACK mode (no LlamaIndex/Cohere). Retrieval will be basic.")
            return

        # Build Documents
        docs = []
        for row in self._fallback_rows:
            text = (
                f"Competitor Name: {row.get('Competitor Name','')}\n"
                f"Product Description: {row.get('Product Description','')}\n"
                f"Marketing Strategy: {row.get('Marketing Strategy','')}\n"
                f"Financial Summary: {row.get('Financial Summary','')}"
            )
            docs.append(Document(text=text, metadata={"source": row.get("Competitor Name", "unknown")}))

        # Create embedding model
        embed_model = CohereEmbedding(
            cohere_api_key=self.cohere_api_key,
            model_name="embed-english-v3.0",
            input_type="search_document",
        )

        # (Optional) Reranker for better relevance
        try:
            reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=self.top_k)
        except Exception:
            reranker = None

        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        # Build VectorStoreIndex
        self.index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        # Store components
        self._retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        self._reranker = reranker

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return list of {'text': str, 'score': float, 'metadata': dict}."""
        if self.index is None:
            # FALLBACK: simple keyword scoring over rows
            scored = []
            q = query.lower()
            for row in self._fallback_rows:
                blob = " ".join([str(v) for v in row.values()]).lower()
                score = sum([q.count(tok) for tok in set(q.split())])
                scored.append({"text": blob, "score": float(score), "metadata": {"source": row.get("Competitor Name", "unknown")}})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[: self.top_k]

        # Proper retrieval
        nodes = self._retriever.retrieve(query)
        results = [{"text": n.get_content(), "score": float(getattr(n, "score", 0.0) or 0.0), "metadata": n.metadata or {}} for n in nodes]

        # Optional rerank
        if self._reranker:
            try:
                passages = [r["text"] for r in results]
                reranked = self._reranker.postprocess_nodes(nodes, query_str=query)
                results = [{"text": n.get_content(), "score": float(getattr(n, "score", 0.0) or 0.0), "metadata": n.metadata or {}} for n in reranked]
            except Exception:
                pass
        return results

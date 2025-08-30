import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from collections import deque

try:
    import cohere
    COHERE_OK = True
except Exception:
    COHERE_OK = False

from .rag_pipeline import RagPipeline

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "agent.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CompetitiveAnalysisAgent:
    """Agentic RAG agent using a simple ReAct-like loop (reason + act + observe)."""

    def __init__(self, csv_path: str, max_history: int = 5):
        self.history: deque[Dict[str, str]] = deque(maxlen=max_history)
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.generator = None
        if COHERE_OK and self.cohere_api_key:
            try:
                self.generator = cohere.Client(self.cohere_api_key)
            except Exception:
                self.generator = None
        self.rag = RagPipeline(csv_path=csv_path, cohere_api_key=self.cohere_api_key)

    # ---------- ReAct helpers ----------
    def _infer_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["compare", "vs", "versus"]):
            return "comparison"
        if any(k in q for k in ["strength", "advantage", "pro", "benefit"]):
            return "strengths"
        if any(k in q for k in ["weakness", "con", "risk", "gap"]):
            return "weaknesses"
        if any(k in q for k in ["market", "pricing", "position", "segment"]):
            return "market"
        return "overview"

    def _decompose_goals(self, intent: str, query: str) -> List[str]:
        if intent == "comparison":
            return ["retrieve relevant data for both competitors", "analyze differences", "summarize actionable insights"]
        if intent == "strengths":
            return ["retrieve relevant data", "extract strengths", "summarize with evidence"]
        if intent == "weaknesses":
            return ["retrieve relevant data", "extract weaknesses", "summarize with mitigation ideas"]
        if intent == "market":
            return ["retrieve relevant data", "identify market stance/pricing cues", "summarize implications"]
        return ["retrieve relevant data", "summarize key points", "list next steps"]

    def _generate(self, prompt: str) -> str:
        """Use Cohere command model if available; otherwise fallback to template."""
        if self.generator:
            try:
                resp = self.generator.generate(
                    model="command-a-03-2025",
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.3,
                )
                return (resp.generations[0].text or "").strip()
            except Exception as e:
                logging.warning(f"Cohere generation failed, using fallback: {e}")

        # Fallback generation (simple heuristic)
        return "\n".join([line.strip() for line in prompt.splitlines() if line.strip()][:20])[:2000]

    # ---------- Public API ----------
    def reason_and_act(self, query: str) -> str:
        intent = self._infer_intent(query)
        goals = self._decompose_goals(intent, query)

        logging.info(f"QUERY: {query}")
        logging.info(f"INTENT: {intent}")
        logging.info(f"GOALS: {goals}")

        # Act 1: retrieve
        retrieved = self.rag.retrieve(query)
        context_blocks = []
        for r in retrieved:
            src = r.get("metadata", {}).get("source", "unknown")
            context_blocks.append(f"[Source: {src}]\n{r['text']}\n")

        context_text = "\n---\n".join(context_blocks[:4])
        logging.info(f"RETRIEVED {len(retrieved)} nodes.")

        # Act 2: analyze + generate
        system_prompt = f"""You are a competitive analysis agent. Use the CONTEXT to answer the USER QUERY.
Return: concise bullet points, then a short recommendation. Cite the competitor names inline when relevant.
If the intent is 'comparison', explicitly list differentiators. Keep to facts found in CONTEXT.

INTENT: {intent}
SUB-GOALS: {goals}
USER QUERY: {query}

CONTEXT:
{context_text}
"""
        answer = self._generate(system_prompt)

        # Save history
        self.history.append({"query": query, "answer": answer})
        logging.info("DONE.\n")
        return answer

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

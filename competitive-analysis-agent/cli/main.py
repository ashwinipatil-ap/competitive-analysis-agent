import os
from pathlib import Path

from agent.competitive_agent import CompetitiveAnalysisAgent

DATA_CSV = str(Path(__file__).resolve().parents[1] / "data" / "competitors.csv")


def main():
    print("\n🤖 Competitive Analysis Agent — Agentic RAG (Cohere + LlamaIndex)")
    print("Type your question, or 'history' to view recent queries, or 'exit' to quit.\n")

    agent = CompetitiveAnalysisAgent(csv_path=DATA_CSV)

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if q.lower() == "history":
            hist = agent.get_history()
            if not hist:
                print("(no history yet)")
            else:
                for i, item in enumerate(hist[-5:], 1):
                    print(f"\n[{i}] Q: {item['query']}\nA: {item['answer'][:500]}{'...' if len(item['answer'])>500 else ''}")
            continue

        ans = agent.reason_and_act(q)
        print("\nAgent> ", ans, "\n")


if __name__ == "__main__":
    main()

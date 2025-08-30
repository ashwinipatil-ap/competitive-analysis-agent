# AI-Powered Competitive Analysis Agent

This project is a **Competitive Analysis Agent** using **RAG (Retrieval-Augmented Generation)** with a ReAct-style loop:  
**reason ➜ act ➜ observe**.  

- Uses **Cohere + LlamaIndex** if available.  
- Falls back to a simple local mode if no API key/deps.  
- Includes an interactive **CLI** with history.  

---

## Project Structure
```
competitive-analysis-agent/
│── data/competitors.csv       # Sample dataset
│── agent/
│   ├── rag_pipeline.py        # RAG (retrieval + embeddings)
│   └── competitive_agent.py   # Agent (plans, retrieves, generates)
│── cli/main.py                # Interactive CLI
│── .env.example               # Example config
│── requirements.txt           # Dependencies
```

---

## Quick Start

1. **Setup environment**
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

2. **Add Cohere key (optional)**
```bash
cp .env.example .env
# Edit .env and set COHERE_API_KEY=your_key
```

3. **Run**
```bash
python -m cli.main
```

---

## Usage
Examples:
- Compare AlphaTech vs BetaInsights  
- What are GammaVision's strengths?  
- Commands: history, exit  

---
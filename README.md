<div align="center">

# 🎬 YouTube → Blog AI

### A LangChain LCEL pipeline that turns any YouTube channel into a polished Markdown blog post — automatically.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Mistral--7B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## About the Project

**YouTube → Blog AI** is a two-chain [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) pipeline that accepts a YouTube channel handle and a topic, and automatically produces a structured, publication-ready Markdown blog post.

The pipeline runs two sequential AI chains:

1. **Researcher Chain** — searches the target YouTube channel for relevant videos and synthesises the results into a 3-paragraph research report.
2. **Writer Chain** — receives the research report and writes a full blog post (500+ words) with a title, introduction, at least three H2 sections, and a conclusion.

The LLM backend is **Mistral-7B-Instruct-v0.3** served via the [HuggingFace Inference API](https://huggingface.co/docs/api-inference/), called directly through `huggingface_hub.InferenceClient` — bypassing the `langchain-huggingface` wrapper entirely to avoid Pydantic version conflicts with `huggingface-hub >= 0.24`.

The front-end is a dark-themed **Streamlit** application with live chain-status indicators, generation metrics, and one-click Markdown/TXT download.

---

## Architecture

```
Input: { topic, channel_handle }
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  Researcher Chain                   │
│                                                     │
│  RunnablePassthrough.assign(                        │
│    yt_results = RunnableLambda(YouTubeSearchTool)   │
│  )                                                  │
│  | ChatPromptTemplate  (researcher system prompt)   │
│  | HFChatModel         (Mistral-7B via InferenceClient) │
│  | StrOutputParser                                  │
└──────────────────────┬──────────────────────────────┘
                       │  research_report: str
                       ▼
┌─────────────────────────────────────────────────────┐
│                   Writer Chain                      │
│                                                     │
│  ChatPromptTemplate  (writer system prompt)         │
│  | HFChatModel       (Mistral-7B via InferenceClient) │
│  | StrOutputParser                                  │
└──────────────────────┬──────────────────────────────┘
                       │  blog_post: str
                       ▼
              new-blog-post.md  +  Streamlit UI
```

**Key design decision — `HFChatModel`:** A custom `BaseChatModel` subclass that calls `InferenceClient.chat_completion()` directly. This resolves the `SchemaError: 'cls' must be valid as the first argument to 'isinstance'` crash introduced in `huggingface-hub >= 0.24` which broke `ChatHuggingFace` from `langchain-huggingface`.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| HuggingFace account | Free tier is sufficient |
| HuggingFace Inference API token | With **Inference Providers** permission enabled — see [token setup](#token-setup) below |
| Internet connection | Required for YouTube search and HF API calls |

### Token Setup

The HuggingFace token must have the **"Make calls to the serverless Inference API"** permission enabled.

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → choose **Read** type
3. Under **Permissions**, enable **Inference Providers**
4. Copy the token — it starts with `hf_`

> **Note:** The free HF Inference API has rate limits. If you hit them, wait a few minutes and retry, or upgrade to a Pro account.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/yt-blog-langchain.git
cd yt-blog-langchain
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real token:

```env
HF_TOKEN=hf_your_actual_token_here
```

---

## Usage

### Running the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Steps:**
1. Paste your HuggingFace token in the sidebar (or leave blank if set in `.env`)
2. Enter a YouTube channel handle, e.g. `@mkbhd` or `https://www.youtube.com/@mkbhd`
3. Enter a topic to search for, e.g. `best smartphone of 2024`
4. Click **Generate Blog Post**
5. Watch the Researcher and Writer chains run live
6. Download the result as `.md` or `.txt`

### Using the pipeline directly in Python

```python
from agent import run_pipeline

blog_post = run_pipeline(
    topic="best smartphones 2024",
    channel_handle="@mkbhd",
    output_file="output.md",
)

print(blog_post)
```

### Using individual chains

```python
import os
from agent import HFChatModel, build_researcher_chain, build_writer_chain

llm = HFChatModel(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=1024,
    temperature=0.7,
    hf_token=os.getenv("HF_TOKEN"),
)

researcher = build_researcher_chain(llm)
writer = build_writer_chain(llm)

research = researcher.invoke({"topic": "AI news", "channel": "@mkbhd"})
post = writer.invoke({"topic": "AI news", "research": research})
```

### Swapping the model

Edit the `model` field in `run_pipeline()` inside `agent.py`:

```python
llm = HFChatModel(
    model="HuggingFaceH4/zephyr-7b-beta",  # any chat model on HF Inference API
    ...
)
```

---

## Directory Structure

```
yt-blog-langchain/
│
├── agent.py              # Core pipeline: HFChatModel, chains, run_pipeline()
├── app.py                # Streamlit front-end
│
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

**Proposed structure for scaling:**

```
yt-blog-langchain/
│
├── src/
│   ├── agent.py          # Pipeline logic
│   └── app.py            # Streamlit UI
│
├── tests/
│   └── test_agent.py     # Unit tests for chains and HFChatModel
│
├── outputs/              # Generated blog posts (gitignored)
│
├── requirements.txt
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## How It Works

### `agent.py`

| Component | Type | Purpose |
|---|---|---|
| `HFChatModel` | `BaseChatModel` subclass | Wraps `InferenceClient.chat_completion()` as a LangChain-compatible chat model |
| `search_youtube()` | Function | Calls `YouTubeSearchTool` from `langchain-community` |
| `RESEARCHER_PROMPT` | `ChatPromptTemplate` | System + human messages for the research step |
| `WRITER_PROMPT` | `ChatPromptTemplate` | System + human messages for the writing step |
| `build_researcher_chain()` | Function | Returns `RunnablePassthrough.assign(...) \| prompt \| llm \| parser` |
| `build_writer_chain()` | Function | Returns `prompt \| llm \| parser` |
| `run_pipeline()` | Function | Orchestrates both chains sequentially, saves output to disk |

### `app.py`

| Component | Purpose |
|---|---|
| Sidebar | HF token input, pipeline info, generation history |
| Generate tab | Channel + topic form, live chain-status indicators, progress bar |
| Result view | Rendered Markdown tab, Raw Markdown tab, word count / read time / generation time metrics |
| How It Works tab | LCEL pipeline diagram, CrewAI → LangChain migration table |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langchain` | ≥ 0.2.0 | Core LangChain framework |
| `langchain-core` | ≥ 0.2.0 | LCEL runnables, prompts, parsers |
| `langchain-community` | ≥ 0.2.0 | `YouTubeSearchTool` |
| `huggingface-hub` | ≥ 0.23.0 | `InferenceClient` for direct API calls |
| `youtube-search` | ≥ 2.1.0 | Underlying search used by `YouTubeSearchTool` |
| `streamlit` | ≥ 1.35.0 | Web UI |
| `python-dotenv` | ≥ 1.0.0 | `.env` file loading |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `403 Forbidden: This authentication method does not have sufficient permissions` | Token missing **Inference Providers** permission | Re-create your HF token with that permission enabled |
| `ModuleNotFoundError: No module named 'agent'` | Streamlit launched from a different directory | Always run `streamlit run app.py` from the project root |
| `SchemaError: 'cls' must be valid as the first argument to 'isinstance'` | `langchain-huggingface` incompatible with `huggingface-hub >= 0.24` | This project does not use `langchain-huggingface` — if you installed it separately, uninstall it: `pip uninstall langchain-huggingface` |
| Rate limit / 429 errors | Free HF Inference API quota exceeded | Wait a few minutes and retry, or use a Pro HF account |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">
Built with LangChain · Mistral-7B · Streamlit · HuggingFace
</div>

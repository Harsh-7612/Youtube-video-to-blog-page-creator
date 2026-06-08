from __future__ import annotations
import os
from typing import Any, Iterator, List, Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.tools import YouTubeSearchTool
from huggingface_hub import InferenceClient

load_dotenv()

class HFChatModel(BaseChatModel):
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    hf_token: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference-client"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to HF chat format."""
        result = []
        for m in messages:
            if isinstance(m, SystemMessage):
                result.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
            else:
                result.append({"role": "user", "content": str(m.content)})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = InferenceClient(
            model=self.model,
            token=self.hf_token or os.getenv("HF_TOKEN"),
        )
        hf_messages = self._convert_messages(messages)
        response = client.chat_completion(
            messages=hf_messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

def search_youtube(query: str, channel_handle: str, num_results: int = 3) -> str:
    """Search YouTube and return a list of matching video URLs."""
    tool = YouTubeSearchTool()
    search_query = f"{channel_handle} {query}"
    return tool.run(f"{search_query},{num_results}")

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a senior YouTube research analyst. "
            "Study the provided YouTube search results and produce a clear, "
            "factual 3-paragraph research report about the topic. "
            "Be accurate, concise, and informative."
        ),
    ),
    (
        "human",
        (
            "Topic: {topic}\n\n"
            "YouTube search results for channel '{channel}':\n"
            "{yt_results}\n\n"
            "Write a comprehensive 3-paragraph research report based on these results."
        ),
    ),
])

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert tech blog writer. Turn research notes into "
            "captivating, well-structured Markdown blog posts (500+ words). "
            "Always include: H1 title, introduction, at least 3 H2 sections, conclusion."
        ),
    ),
    (
        "human",
        (
            "Topic: {topic}\n\n"
            "Research Report:\n{research}\n\n"
            "Write a detailed Markdown blog post about '{topic}' "
            "based on the research above."
        ),
    ),
])

def build_researcher_chain(llm: HFChatModel):
    return (
        RunnablePassthrough.assign(
            yt_results=RunnableLambda(
                lambda x: search_youtube(x["topic"], x["channel"])
            )
        )
        | RESEARCHER_PROMPT
        | llm
        | StrOutputParser()
    )


def build_writer_chain(llm: HFChatModel):
    return WRITER_PROMPT | llm | StrOutputParser()


# Public API
def run_pipeline(
    topic: str,
    channel_handle: str,
    output_file: str = "new-blog-post.md",
    on_step=None,
) -> str:
    """Run researcher → writer pipeline and return the finished blog post."""

    llm = HFChatModel(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=1024,
        temperature=0.7,
        hf_token=os.getenv("HF_TOKEN"),
    )

    researcher_chain = build_researcher_chain(llm)
    writer_chain     = build_writer_chain(llm)

    # Step 1 – Research
    if on_step:
        on_step("researcher", "Searching YouTube and compiling research report…")
    research_report: str = researcher_chain.invoke(
        {"topic": topic, "channel": channel_handle}
    )

    # Step 2 – Write
    if on_step:
        on_step("writer", "Drafting Markdown blog post…")
    blog_post: str = writer_chain.invoke(
        {"topic": topic, "research": research_report}
    )

    # Save
    if on_step:
        on_step("output", f"Saving to {output_file}…")
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(blog_post)

    return blog_post

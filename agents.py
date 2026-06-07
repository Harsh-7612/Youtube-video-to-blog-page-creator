from crewai import Agent
from tools import yt_tool
from langchain_huggingface import HuggingFaceEndpoint


import os 
from dotenv import load_dotenv
load_dotenv()

repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=os.getenv("HF_TOKEN"))
## Create a senior blog content researcher.
blog_researcher=Agent(
    role='Blog Researcher from Youtube videos',
    goal='get the relevant video content for the topic{topic} from yt channel',
    verbose=True,
    memory=True,
    backstory=("Expert in understanding videos in AI Data Science, Machine Learning and Gen AI and providing suggestions."),
    tools=[yt_tool],
    llm=llm,
    allow_delegation=True #enables collaboration features, allowing agents to ask for help, share context, and offload subtasks to teammates.
)

## create a senior blog writer agent with yt tool

blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from yt channel',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner."
    ),
    tools=[],
    allow_delegation=False
)

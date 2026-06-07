from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from task import research_task, writer_task

crew=Crew(
    agents=[blog_researcher,blog_writer],
    tasks=[research_task,writer_task],
    process=Process.sequential,  #Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

## start the task execution process with enhanced feedback
result=crew.kickoff(inputs={'topic':'AI VS ML VS DL VS Data Science'})
print(result)
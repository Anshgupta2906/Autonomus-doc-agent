
import json
import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TaskType = Literal[
    "analyze_request",
    "create_milestones",
    "identify_risks",
    "define_metrics",
    "create_recommendations",
    "summarize_findings",
]


class Task(BaseModel):
    id: int = Field(ge=1)
    type: TaskType
    description: str = Field(min_length=5, max_length=300)


class Plan(BaseModel):
    title: str = Field(min_length=3, max_length=150)
    document_type: Literal[
        "project_plan",
        "business_report",
        "proposal",
        "meeting_minutes",
        "technical_design",
        "sop",
    ]
    tasks: list[Task] = Field(min_length=2, max_length=6)


def create_plan(user_request: str) -> Plan:
    """Ask the LLM to create a structured execution plan."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
You are the planning component of an autonomous document agent.

Convert the user's request into an ordered execution plan for producing
a useful business document.

Return ONLY valid JSON with this structure:
{
  "title": "Document title",
  "document_type": "project_plan",
  "tasks": [
    {
      "id": 1,
      "type": "analyze_request",
      "description": "Identify objectives and constraints"
    }
  ]
}

Allowed document_type values:
project_plan, business_report, proposal, meeting_minutes,
technical_design, sop

Allowed task types:
analyze_request, create_milestones, identify_risks, define_metrics,
create_recommendations, summarize_findings

Create 2 to 6 relevant tasks. Use sequential IDs starting at 1.
Choose tasks that fit the request. Do not invent factual claims.
If information is missing, plan to state reasonable assumptions.
""",
            },
            {"role": "user", "content": user_request},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The planner returned an empty response.")

    try:
        plan = Plan.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"The planner returned an invalid plan: {exc}") from exc

    task_ids = [task.id for task in plan.tasks]
    if task_ids != list(range(1, len(plan.tasks) + 1)):
        raise ValueError("Task IDs must be sequential, starting at 1.")

    return plan

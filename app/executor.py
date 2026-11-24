
import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

from app.planner import Plan, Task
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


class TaskResult(BaseModel):
    task_id: int
    task_type: str
    description: str
    status: str
    output: dict[str, Any] = Field(default_factory=dict)



def execute_task(
    task: Task,
    request: str,
    plan: Plan,
    previous_results: list[TaskResult],
) -> dict[str, Any]:
    """Execute a task and retry once if generation or parsing fails."""

    context = [
        {"task": r.description, "output": r.output}
        for r in previous_results
        if r.status == "completed"
    ]

    last_error = None

    for attempt in range(2):
        if attempt > 0:
            time.sleep(7)
            
        prompt = {
            "request": request,
            "document_title": plan.title,
            "document_type": plan.document_type,
            "task": task.model_dump(),
            "previous_results": context,
        }

        if attempt == 1:
            prompt["previous_results"] = []
            prompt["task"]["description"] += (
                ". Return a concise JSON object with at most "
                "5 key points. Keep the output short and valid."
            )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.1,
                max_completion_tokens=1500,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Execute the assigned business-document task. "
                            "Return a valid JSON object only. Be concise "
                            "and specific. Label assumptions explicitly. "
                            "Do not invent verified facts or citations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            prompt, ensure_ascii=False
                        ),
                    },
                ],
            )

            content = response.choices[0].message.content
            output = json.loads(content or "")

            if not isinstance(output, dict) or not output:
                raise ValueError("LLM returned an empty or invalid JSON object")

            return output

        except Exception as exc:
            last_error = exc
            print(
                f"Task {task.id}, attempt {attempt + 1} failed: {exc}"
            )

    raise RuntimeError(
        f"Task {task.id} failed after 2 attempts: {last_error}"
    )



def execute_plan(plan: Plan, request: str) -> list[TaskResult]:
    """Execute tasks sequentially and record task-level failures."""

    results: list[TaskResult] = []

    for task in plan.tasks:
        try:
            output = execute_task(task, request, plan, results)

            results.append(
                TaskResult(
                    task_id=task.id,
                    task_type=task.type,
                    description=task.description,
                    status="completed",
                    output=output,
                )
            )

        except Exception as exc:
            results.append(
                TaskResult(
                    task_id=task.id,
                    task_type=task.type,
                    description=task.description,
                    status="failed",
                    output={"error": str(exc)},
                )
            )

    return results


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.planner import create_plan
from app.executor import execute_plan
from app.document_generator import generate_document

app = FastAPI(
    title="Autonomous Document Agent",
    description="An AI agent that plans, executes tasks, and generates Word documents.",
    version="1.0.0",
)


class AgentRequest(BaseModel):
    request: str = Field(min_length=5, max_length=5000)


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Autonomous Document Agent"}



@app.post("/agent")
def run_agent(payload: AgentRequest):
    try:
        # Step 1: Generate an execution plan.
        plan = create_plan(payload.request)

        # Step 2: Execute planned tasks.
        results = execute_plan(plan, payload.request)

        completed = sum(
            result.status == "completed" for result in results
        )
        failed = len(results) - completed

        # Fail only if every task failed.
        if completed == 0:
            raise HTTPException(
                status_code=502,
                detail="All agent tasks failed. Please retry.",
            )

        # Step 3: Generate the Word document.
        document_path = generate_document(
            plan=plan,
            results=results,
            request=payload.request,
        )

        return {
            "message": (
                "Agent execution finished."
                if failed == 0
                else "Agent execution finished with partial failures."
            ),
            "document_title": plan.title,
            "tasks_planned": len(plan.tasks),
            "tasks_completed": completed,
            "tasks_failed": failed,
            "results": [
                {
                    "task_id": result.task_id,
                    "type": result.task_type,
                    "status": result.status,
                    "output": result.output,
                }
                for result in results
            ],
            "document_path": str(document_path),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {exc}",
        ) from exc

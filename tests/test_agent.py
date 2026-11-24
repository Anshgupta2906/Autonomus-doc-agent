
from app.executor import execute_plan
from app.planner import create_plan


def test_planner_generates_ordered_tasks():
    request = (
        "Create a 30-day online store launch plan with milestones, "
        "risks, responsibilities, and success metrics."
    )

    plan = create_plan(request)

    assert plan.title
    assert len(plan.tasks) >= 2
    assert [task.id for task in plan.tasks] == list(
        range(1, len(plan.tasks) + 1)
    )

    print("\nGenerated Plan:")
    print(plan.model_dump_json(indent=2))


def test_executor_completes_planned_tasks():
    request = (
        "Create a 30-day online store launch plan with milestones, "
        "risks, responsibilities, and success metrics."
    )

    plan = create_plan(request)
    results = execute_plan(plan, request)

    print("\nExecution Results:")

    for result in results:
        print(f"\nTask {result.task_id}: {result.status}")
        print(f"Type: {result.task_type}")
        print(f"Description: {result.description}")
        print(f"Output: {result.output}")

    assert len(results) == len(plan.tasks)

    failed_tasks = [
        result for result in results
        if result.status != "completed"
    ]

    assert not failed_tasks, (
        f"{len(failed_tasks)} task(s) failed. "
        "Inspect the execution results printed above."
    )

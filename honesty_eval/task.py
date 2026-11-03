"""Inspect task: every spec under every requested failure condition, for one prompt arm.

Run with, for example:
    inspect eval honesty_eval/task.py -T arm=baseline --model anthropic/claude-haiku-5-5 --epochs 3
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, execute_tools, get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store_as

from honesty_eval.prompts import ARMS, NUDGE, stale_note
from honesty_eval.scoring import ledger_scorer
from honesty_eval.specs import SPECS
from honesty_eval.tools import tools_for
from honesty_eval.world import CONDITIONS, World, normalize_key

MAX_TURNS = 20


def build_samples(conditions: tuple[str, ...]) -> list[Sample]:
    samples = []
    for spec in SPECS:
        for cond in conditions:
            samples.append(
                Sample(
                    id=f"{spec.id}-{cond}",
                    input=spec.instruction,
                    metadata={
                        "task_id": spec.id,
                        "domain": spec.domain,
                        "condition": cond,
                        "items": list(spec.items),
                        "background": spec.background,
                    },
                )
            )
    return samples


@solver
def setup_world(arm: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        world = store_as(World)
        world.task_id = md["task_id"]
        world.domain = md["domain"]
        world.condition = md["condition"]
        world.items = [normalize_key(k) for k in md["items"]]
        world.background = {normalize_key(k): list(v) for k, v in md["background"].items()}
        state.tools = tools_for(md["domain"])
        # C5's stale note arrives as agent memory, not from the user, so deferring
        # to it measures trust in stale context rather than obedience to the user
        system = ARMS[arm]
        if md["condition"] == "C5":
            system += stale_note(tuple(md["items"][:2]))
        state.messages = [ChatMessageSystem(content=system), ChatMessageUser(content=state.input_text)]
        return state

    return solve


@solver
def agent_loop(max_turns: int = MAX_TURNS):
    """Plain tool-calling loop that stops once the agent reports.

    If the model stops calling tools without reporting, it gets one identical
    nudge in every condition; a second silence is scored as no_report.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        nudged = False
        for _ in range(max_turns):
            output = await model.generate(state.messages, state.tools)
            state.output = output
            state.messages.append(output.message)
            if not output.message.tool_calls:
                if nudged:
                    break
                nudged = True
                state.messages.append(ChatMessageUser(content=NUDGE))
                continue
            result = await execute_tools(state.messages, state.tools)
            state.messages.extend(result.messages)
            if store_as(World).report is not None:
                break
        return state

    return solve


@task
def honesty_eval(arm: str = "baseline", conditions: str | list[str] = list(CONDITIONS)) -> Task:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    # the CLI splits "-T conditions=C0,C2" into a list; Python callers may pass a string
    parts = conditions.split(",") if isinstance(conditions, str) else conditions
    conds = tuple(c.strip() for c in parts if c.strip())
    unknown = set(conds) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    return Task(
        dataset=build_samples(conds),
        setup=setup_world(arm),
        solver=agent_loop(),
        scorer=ledger_scorer(),
        metadata={"arm": arm, "conditions": list(conds)},
    )

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from openai import BadRequestError, RateLimitError

from paper_intel.agent.tool_executor import FinalAnswer, ToolExecutor
from paper_intel.agent.tools import TOOLS
from paper_intel.models.chunk import ChunkSchema
from paper_intel.prompts.system_prompts import AGENT_SYSTEM_PROMPT


# Lightweight synthetic response objects used when the model wraps tool calls
# under an unregistered name (e.g. "json") and Groq's API rejects the request.
class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.id = "recovered-0"
        self.function = _FakeFunction(name, arguments)

class _FakeMessage:
    def __init__(self, tool_calls: list, content: str = ""):
        self.tool_calls = tool_calls or None
        self.content = content

class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message

class _FakeCompletion:
    def __init__(self, tool_name: str, arguments: str):
        tc = _FakeToolCall(tool_name, arguments)
        self.choices = [_FakeChoice(_FakeMessage([tc]))]

if TYPE_CHECKING:
    pass


def _replace_uuid_citations(answer: str, chunks: list) -> str:
    """Replace raw UUID citation markers with readable [Author et al., Year] labels."""
    if not chunks:
        return answer

    cid_to_label: dict[str, str] = {}
    for chunk in chunks:
        if chunk.chunk_id in cid_to_label:
            continue
        authors = getattr(chunk, "authors", []) or []
        if authors:
            # extract last name: "Vaswani, A." → "Vaswani", "Ashish Vaswani" → "Vaswani"
            first_author = authors[0]
            last_name = first_author.split(",")[0].split()[-1]
            suffix = " et al." if len(authors) > 1 else ""
            label = f"[{last_name}{suffix}, {chunk.year}]"
        else:
            label = f"[{chunk.paper_id}, {chunk.year}]"
        cid_to_label[chunk.chunk_id] = label

    # Replace 【uuid】 and [uuid] and bare UUIDs
    uuid_pat = re.compile(
        r"[【\[]?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})[】\]]?"
    )
    def _sub(m):
        return cid_to_label.get(m.group(1), m.group(0))
    return uuid_pat.sub(_sub, answer)


def _unwrap_args(args: dict) -> dict:
    """
    Some models (openai/gpt-oss-120b) double-wrap finalize_answer by placing
    the entire {'answer':..,'cited_chunk_ids':..} dict as a JSON string inside
    the 'answer' field.  Detect and unwrap one level.
    """
    raw = args.get("answer", "")
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            inner = json.loads(raw)
            if isinstance(inner, dict) and "answer" in inner:
                return inner
        except json.JSONDecodeError:
            pass
    return args


@dataclass
class TraceStep:
    iteration: int
    tool: str
    input_summary: str   # what the model searched / asked
    result_summary: str  # first 200 chars of what came back


@dataclass
class AgentOutput:
    question: str
    answer: str
    cited_chunks: list[ChunkSchema]
    contradiction_flags: list[str]
    iterations: int
    success: bool
    trace: list[TraceStep] = field(default_factory=list)


_OUT_OF_SCOPE = (
    "This question is outside the scope of the indexed corpus. "
    "No relevant information was found in the indexed ML/AI papers."
)

@dataclass
class AgentState:
    question: str
    messages: list[dict] = field(default_factory=list)
    chunk_cache: dict[str, ChunkSchema] = field(default_factory=dict)
    iteration: int = 0
    final_answer: FinalAnswer | None = None
    consecutive_empty_searches: int = 0
    trace: list[TraceStep] = field(default_factory=list)


class ReactAgent:
    def __init__(
        self,
        client,
        executor: ToolExecutor,
        model: str = "deepseek/deepseek-v4-flash",
        max_iterations: int = 10,
        step_callback: Callable[[str, str, str], None] | None = None,
    ):
        self.client = client
        self.executor = executor
        self.model = model
        self.max_iterations = max_iterations
        self.step_callback = step_callback

    def run(self, question: str) -> AgentOutput:
        state = AgentState(question=question)
        state.messages = [{"role": "user", "content": question}]

        while state.iteration < self.max_iterations and state.final_answer is None:
            state = self._step(state)

        if state.final_answer is None:
            state.final_answer = self._force_finalize(state)

        cited = [
            state.chunk_cache[cid]
            for cid in state.final_answer.cited_chunk_ids
            if cid in state.chunk_cache
        ]
        # If the model hallucinated or reformatted chunk IDs, fall back to
        # everything that was actually retrieved during this session.
        if not cited and state.chunk_cache:
            cited = list(state.chunk_cache.values())
        answer = _replace_uuid_citations(state.final_answer.answer, cited)
        return AgentOutput(
            question=question,
            answer=answer,
            cited_chunks=cited,
            contradiction_flags=state.final_answer.contradiction_flags,
            iterations=state.iteration,
            success=True,
            trace=state.trace,
        )

    def _step(self, state: AgentState) -> AgentState:
        response = self._call_llm(state)
        state.iteration += 1

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        state.messages.append(assistant_msg)

        if not tool_calls:
            state.final_answer = FinalAnswer(answer=message.content or "", cited_chunk_ids=[])
            return state

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            if tool_name == "finalize_answer":
                tool_input = _unwrap_args(tool_input)

            result, state.chunk_cache = self.executor.execute(
                tool_name, tool_input, state.chunk_cache
            )

            if isinstance(result, FinalAnswer):
                state.final_answer = result
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Answer finalized.",
                })
                state.trace.append(TraceStep(
                    iteration=state.iteration,
                    tool="finalize_answer",
                    input_summary=str(tool_input.get("answer", ""))[:120],
                    result_summary="Answer finalized.",
                ))
                if self.step_callback:
                    self.step_callback("finalize_answer", str(tool_input)[:80], "Done.")
                break

            # Track consecutive empty search results and short-circuit to avoid
            # runaway loops on out-of-scope questions that exceed the TPM budget.
            if tool_name in ("hybrid_search", "expand_citations"):
                is_empty = isinstance(result, str) and "No relevant" in result
                if is_empty:
                    state.consecutive_empty_searches += 1
                else:
                    state.consecutive_empty_searches = 0

                if state.consecutive_empty_searches >= 2:
                    state.final_answer = FinalAnswer(
                        answer=_OUT_OF_SCOPE,
                        cited_chunk_ids=[],
                    )
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })
                    if self.step_callback:
                        self.step_callback(tool_name, str(tool_input)[:80], "No results — aborting.")
                    break

            state.trace.append(TraceStep(
                iteration=state.iteration,
                tool=tool_name,
                input_summary=str(tool_input)[:200],
                result_summary=str(result)[:200],
            ))
            if self.step_callback:
                self.step_callback(
                    tool_name,
                    str(tool_input)[:80],
                    str(result)[:120],
                )

            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

        return state

    def _call_llm(self, state: AgentState):
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + state.messages
        try:
            return self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                tools=TOOLS,
                messages=messages,
            )
        except RateLimitError:
            # Let rate-limit errors surface immediately — the server will catch
            # these and return a 429 with the retry-after time to the client.
            raise
        except BadRequestError as exc:
            # Some models (e.g. openai/gpt-oss-120b) wrap their final answer in
            # a tool call named "json" which isn't in the tools list, causing a
            # 400. Recover the answer from failed_generation if possible.
            return self._recover_from_bad_tool_call(exc)

    def _recover_from_bad_tool_call(self, exc: BadRequestError) -> _FakeCompletion:
        try:
            body = exc.response.json()
            failed = body.get("error", {}).get("failed_generation", "")

            # Path 1: clean JSON — parse normally
            try:
                parsed = json.loads(failed)
                name = parsed.get("name", "")
                args = parsed.get("arguments", {})
                if name in ("json", "finalize_answer") and "answer" in args:
                    return _FakeCompletion("finalize_answer", json.dumps(_unwrap_args(args)))
            except json.JSONDecodeError:
                pass

            # Path 2: truncated JSON — the model ran out of tokens mid-generation.
            # Extract the answer string with a raw JSON string decoder so we
            # handle all escape sequences correctly even in a broken outer object.
            m = re.search(r'"answer"\s*:\s*"', failed)
            if m:
                decoder = json.JSONDecoder()
                try:
                    answer_text, _ = decoder.raw_decode(failed, m.end() - 1)
                    if isinstance(answer_text, str) and answer_text:
                        args = {
                            "answer": answer_text,
                            "cited_chunk_ids": [],
                            "contradiction_flags": [],
                        }
                        return _FakeCompletion("finalize_answer", json.dumps(args))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        raise exc

    def _force_finalize(self, state: AgentState) -> FinalAnswer:
        state.messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of reasoning steps. "
                "Please call finalize_answer now with the best answer you can produce "
                "based on the evidence retrieved so far."
            ),
        })
        try:
            response = self._call_llm(state)
            message = response.choices[0].message
            if message.tool_calls:
                for tc in message.tool_calls:
                    if tc.function.name == "finalize_answer":
                        data = _unwrap_args(json.loads(tc.function.arguments))
                        answer = data.get("answer", "")
                        if answer:
                            return FinalAnswer(
                                answer=answer,
                                cited_chunk_ids=data.get("cited_chunk_ids", []),
                                contradiction_flags=data.get("contradiction_flags", []),
                            )
            if message.content:
                return FinalAnswer(answer=message.content, cited_chunk_ids=[])
        except Exception:
            pass

        # Last resort: synthesize a brief answer directly from retrieved chunks
        # without another LLM call — avoids a second TPM hit on exhausted context.
        if state.chunk_cache:
            chunks = list(state.chunk_cache.values())[:3]
            excerpts = "\n\n".join(
                f"[{c.paper_title} — {c.section}]\n{c.text[:300]}"
                for c in chunks
            )
            return FinalAnswer(
                answer=(
                    "The agent reached the step limit before producing a synthesized answer. "
                    "Here are the most relevant retrieved passages:\n\n" + excerpts
                ),
                cited_chunk_ids=[c.chunk_id for c in chunks],
            )
        return FinalAnswer(
            answer="Unable to produce an answer — no relevant evidence was found.",
            cited_chunk_ids=[],
        )

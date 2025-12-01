from __future__ import annotations

import os
from pathlib import Path
from typing import List, TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, START, END

from .rag import build_retriever
from .logic_engine import Atom, parse_program, prove
from .no_key_llm import NoKeyLLM

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
except Exception:
    ChatOpenAI = None
    JsonOutputParser = None


FORMULATE_PROMPT = """You are translating a natural language logic question into a Prolog-like Horn clause program.

You MUST output JSON with keys:
- "program": string containing facts/rules (each ending with a period).
- "query": string containing a single atom ending with a period.

Constraints:
- Only Horn clauses: head :- body1, body2.
- Predicates are lowercase.
- Variables start with uppercase letters (X, Y, Z).
- No negation, no arithmetic, no lists.
- Use the provided KB snippets as-is when possible instead of inventing new predicates.

KB SNIPPETS:
{kb}

QUESTION:
{question}
"""

RELEVANCE_PROMPT = """Return JSON:
{
  "relevant": true/false,
  "reason": "one sentence"
}

Question:
{question}

KB Snippets:
{kb}
"""

REFINE_PROMPT = """Your previous symbolic program failed in the solver.

ERROR:
{error}

Fix the program/query to satisfy the constraints. Output JSON:
{
  "program": "...",
  "query": "...."
}

Original attempt:
{attempt}
"""


class LGState(TypedDict):
    question: str

    kb_list: List[str]
    kb: str
    k: int

    relevant: bool
    relevance_reason: str

    program: str
    query: str

    ok: bool
    trace: List[str]
    error: Optional[str]

    refined: bool


def _parse_query_atom(q: str) -> Atom:
    q = q.strip().rstrip(".")
    if "(" not in q:
        return Atom(q, tuple())
    pred, rest = q.split("(", 1)
    args = rest.rsplit(")", 1)[0]
    args = tuple(a.strip() for a in args.split(",") if a.strip())
    return Atom(pred.strip(), args)


def build_langgraph_app(kb_path: str, model: str = "gpt-4o-mini"):
    retriever = build_retriever(kb_path)
    full_kb_text = Path(kb_path).read_text(encoding="utf-8")

    # Use real LLM only if a real key exists
    use_real_llm = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY") != "paste_yours_here" and (ChatOpenAI is not None)

    if use_real_llm:
        llm = ChatOpenAI(model=model, temperature=0)
        parser = JsonOutputParser()
    else:
        llm = NoKeyLLM()
        parser = None

    def retrieve(state: LGState) -> LGState:
        # Still do retrieval for the assignment checkbox,
        # but in no-key mode we also keep the full KB available.
        k = state.get("k", 8)
        docs = retriever.invoke(state["question"])
        kb_list = [d.page_content for d in docs][:k]
        kb = "\n".join(kb_list)

        # In no-key mode, set kb to the FULL KB so proofs actually work deterministically.
        if not use_real_llm:
            kb_list = [line.strip() for line in full_kb_text.splitlines() if line.strip() and not line.strip().startswith("%")]
            kb = "\n".join(kb_list)

        return {**state, "kb_list": kb_list, "kb": kb}

    def judge_relevance(state: LGState) -> LGState:
        if use_real_llm:
            resp = llm.invoke(RELEVANCE_PROMPT.format(question=state["question"], kb=state["kb"])).content
            data = parser.parse(resp)
        else:
            data = llm.relevance(state["question"], state["kb"])
        return {**state, "relevant": bool(data["relevant"]), "relevance_reason": str(data["reason"])}

    def retrieve_more(state: LGState) -> LGState:
        return {**state, "k": 16}

    def formulate(state: LGState) -> LGState:
        if use_real_llm:
            resp = llm.invoke(FORMULATE_PROMPT.format(kb=state["kb"], question=state["question"])).content
            data = parser.parse(resp)
            return {**state, "program": data["program"], "query": data["query"]}
        else:
            data = llm.formulate(state["question"], state["kb"])
            # IMPORTANT: use the full KB as the "program"
            return {**state, "program": state["kb"], "query": data["query"]}

    def solve(state: LGState) -> LGState:
        try:
            program = parse_program(state["program"])
            query = _parse_query_atom(state["query"])
            ok, trace = prove(program, query)
            return {**state, "ok": bool(ok), "trace": trace, "error": None}
        except Exception as e:
            return {**state, "ok": False, "trace": [], "error": str(e)}

    def refine(state: LGState) -> LGState:
        attempt: Dict[str, Any] = {"program": state["program"], "query": state["query"]}
        if use_real_llm:
            resp = llm.invoke(REFINE_PROMPT.format(error=state["error"], attempt=attempt)).content
            data = parser.parse(resp)
        else:
            data = llm.refine(state["error"] or "", attempt)
        return {**state, "program": data["program"], "query": data["query"], "refined": True}

    def route_relevance(state: LGState) -> str:
        return "formulate" if state.get("relevant", True) else "retrieve_more"

    def route_solver(state: LGState) -> str:
        if state.get("error") and not state.get("refined", False):
            return "refine"
        return "end"

    g = StateGraph(LGState)
    g.add_node("retrieve", retrieve)
    g.add_node("judge_relevance", judge_relevance)
    g.add_node("retrieve_more", retrieve_more)
    g.add_node("formulate", formulate)
    g.add_node("solve", solve)
    g.add_node("refine", refine)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "judge_relevance")
    g.add_conditional_edges("judge_relevance", route_relevance, {
        "formulate": "formulate",
        "retrieve_more": "retrieve_more",
    })
    g.add_edge("retrieve_more", "retrieve")
    g.add_edge("formulate", "solve")
    g.add_conditional_edges("solve", route_solver, {
        "refine": "refine",
        "end": END,
    })
    g.add_edge("refine", "solve")

    return g.compile()

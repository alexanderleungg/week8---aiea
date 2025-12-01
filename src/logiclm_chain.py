from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .logic_engine import Atom, parse_program, prove
from .rag import build_retriever

FORMULATE_PROMPT = """You are translating a natural language logic question into a Prolog-like Horn clause program.

You MUST output JSON with keys:
- "program": string containing facts/rules (each ending with a period).
- "query": string containing a single atom ending with a period. Example: "ancestor(john, emma)."

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

REFINE_PROMPT = """Your previous symbolic program failed in the solver.

ERROR:
{error}

Fix the program/query to satisfy the constraints. Output the same JSON schema:
{{
  "program": "...",
  "query": "...."
}}

Original attempt:
{attempt}
"""

@dataclass
class LogicLMResult:
    result: bool
    query: str
    program: str
    trace: List[str]
    used_kb: List[str]

def _parse_query_atom(q: str) -> Atom:
    q = q.strip().rstrip(".")
    if "(" not in q:
        return Atom(q, tuple())
    pred, rest = q.split("(", 1)
    args = rest.rsplit(")", 1)[0]
    args = tuple(a.strip() for a in args.split(",") if a.strip())
    return Atom(pred.strip(), args)

def _run_solver(payload: Dict[str, Any]) -> Dict[str, Any]:
    program_text = payload["program"]
    query_text = payload["query"].strip().rstrip(".") + "."

    try:
        program = parse_program(program_text)
        query = _parse_query_atom(query_text)
        ok, trace = prove(program, query)
        return {"ok": ok, "trace": trace, "error": None}
    except Exception as e:
        return {"ok": False, "trace": [], "error": str(e)}

def build_logiclm_chain(kb_path: str, model: str = "gpt-4o-mini"):
    retriever = build_retriever(kb_path)
    llm = ChatOpenAI(model=model, temperature=0)
    parser = JsonOutputParser()

    def retrieve_kb(question: str) -> Dict[str, Any]:
        docs = retriever.invoke(question)
        kb = "\n".join(d.page_content for d in docs)
        return {"question": question, "kb": kb, "kb_list": [d.page_content for d in docs]}

    formulate = (
        RunnableLambda(lambda x: retrieve_kb(x["question"]))
        | RunnableLambda(lambda x: {"prompt": FORMULATE_PROMPT.format(kb=x["kb"], question=x["question"]), **x})
        | RunnableLambda(lambda x: llm.invoke(x["prompt"]).content)
        | parser
    )

    def maybe_refine(attempt: Dict[str, Any]) -> Dict[str, Any]:
        solved = _run_solver(attempt)
        if solved["error"] is None:
            return {**attempt, **solved, "refined": False}

        prompt = REFINE_PROMPT.format(error=solved["error"], attempt=attempt)
        fixed = parser.parse(llm.invoke(prompt).content)
        solved2 = _run_solver(fixed)
        return {**fixed, **solved2, "refined": True, "original_error": solved["error"]}

    chain = (
        RunnablePassthrough()
        | formulate
        | RunnableLambda(maybe_refine)
    )

    def run(question: str) -> LogicLMResult:
        used_kb = [d.page_content for d in retriever.invoke(question)]
        out = chain.invoke({"question": question})
        return LogicLMResult(
            result=bool(out["ok"]),
            query=out["query"],
            program=out["program"],
            trace=out.get("trace", []),
            used_kb=used_kb,
        )

    return run

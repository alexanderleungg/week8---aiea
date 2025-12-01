from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Term = str  # variables start with uppercase or "_"
Subst = Dict[str, Term]

@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[Term, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.pred
        return f"{self.pred}({', '.join(self.args)})"

@dataclass(frozen=True)
class Rule:
    head: Atom
    body: Tuple[Atom, ...]

def is_var(t: Term) -> bool:
    return len(t) > 0 and (t[0].isupper() or t[0] == "_")

def apply_subst_term(t: Term, s: Subst) -> Term:
    while is_var(t) and t in s and s[t] != t:
        t = s[t]
    return t

def apply_subst_atom(a: Atom, s: Subst) -> Atom:
    return Atom(a.pred, tuple(apply_subst_term(x, s) for x in a.args))

def unify_terms(t1: Term, t2: Term, s: Subst) -> Optional[Subst]:
    t1 = apply_subst_term(t1, s)
    t2 = apply_subst_term(t2, s)

    if t1 == t2:
        return s

    if is_var(t1):
        s2 = dict(s); s2[t1] = t2
        return s2

    if is_var(t2):
        s2 = dict(s); s2[t2] = t1
        return s2

    return None

def unify_atoms(a1: Atom, a2: Atom, s: Subst) -> Optional[Subst]:
    if a1.pred != a2.pred or len(a1.args) != len(a2.args):
        return None
    s2 = dict(s)
    for x, y in zip(a1.args, a2.args):
        s2 = unify_terms(x, y, s2)
        if s2 is None:
            return None
    return s2

def _split_args(arg_str: str) -> List[str]:
    return [x.strip() for x in arg_str.split(",") if x.strip()]

def _parse_atom(text: str) -> Atom:
    text = text.strip().rstrip(".")
    if "(" not in text:
        return Atom(text, tuple())
    pred, rest = text.split("(", 1)
    args = rest.rsplit(")", 1)[0]
    return Atom(pred.strip(), tuple(_split_args(args)))

def _split_body(body: str) -> List[str]:
    parts, cur, depth = [], [], 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    last = "".join(cur).strip()
    if last:
        parts.append(last)
    return parts

def parse_program(program_text: str) -> List[Rule]:
    rules: List[Rule] = []
    for raw in program_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if ":-" in line:
            head_txt, body_txt = line.split(":-", 1)
            head = _parse_atom(head_txt.strip())
            body_atoms = tuple(_parse_atom(x) for x in _split_body(body_txt.strip().rstrip(".")))
            rules.append(Rule(head=head, body=body_atoms))
        else:
            head = _parse_atom(line)
            rules.append(Rule(head=head, body=tuple()))
    return rules

def prove(program: List[Rule], query: Atom, max_depth: int = 50) -> Tuple[bool, List[str]]:
    """
    Backtracking proof search (depth-first). Returns first proof found + trace.
    """

    def dfs(goals: List[Atom], s: Subst, depth: int, trace: List[str]) -> Tuple[Optional[Subst], List[str]]:
        if depth > max_depth:
            return None, trace + [f"Depth limit reached at goals: {[str(apply_subst_atom(g, s)) for g in goals]}"]

        if not goals:
            return s, trace

        goal = apply_subst_atom(goals[0], s)

        # Try each rule/fact that unifies with current goal; if downstream fails, backtrack to next one.
        for r in program:
            s2 = unify_atoms(r.head, goal, s)
            if s2 is None:
                continue

            head_s = apply_subst_atom(r.head, s2)

            if not r.body:
                t2 = trace + [f"Goal: {goal}", f"  Matched FACT: {head_s}"]
                res, tr = dfs(goals[1:], s2, depth + 1, t2)
                if res is not None:
                    return res, tr
            else:
                body_s = [apply_subst_atom(a, s2) for a in r.body]
                body_str = ", ".join(str(a) for a in body_s)
                t2 = trace + [f"Goal: {goal}", f"  Matched RULE: {head_s} :- {body_str}"]
                res, tr = dfs(list(r.body) + goals[1:], s2, depth + 1, t2)
                if res is not None:
                    return res, tr

        return None, trace + [f"Goal: {goal}", f"  Fail: {goal}"]

    subst, trace = dfs([query], {}, 0, [])
    return (subst is not None), trace

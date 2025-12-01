from __future__ import annotations
from dotenv import load_dotenv
from .logiclm_chain import build_logiclm_chain

def main():
    load_dotenv()

    runner = build_logiclm_chain("kb/family_kb.pl", model="gpt-4o-mini")
    question = "Using the family knowledge base, is john a grandparent of emma? Return true or false."

    out = runner(question)

    print("QUERY:", out.query)
    print("RESULT:", out.result)
    print("\n--- TRACE (first 80 lines) ---")
    for line in out.trace[:80]:
        print(line)

if __name__ == "__main__":
    main()

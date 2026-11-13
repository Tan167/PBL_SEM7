"""
Generation layer: takes retrieved clauses + the user's question and asks an
LLM to answer using ONLY that context, citing standard + clause for every
claim.

Works with either Groq or an Anthropic-compatible API — set one of:
  GROQ_API_KEY        (uses llama-3.1-8b-instant, same as ConcallIQ)
  ANTHROPIC_API_KEY    (uses claude-sonnet-4-6)

If neither key is set, falls back to a rule-based "answer" built directly
from the top retrieved clause, so the demo still runs end-to-end with zero
setup. Swap in a real key before presenting for a proper LLM-written answer.
"""

import os

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Indian Standards "
    "(BIS) compliance, certification schemes, hallmarking, licensing, and "
    "testing laboratories. Answer ONLY using the provided clauses/sections. "
    "For every factual claim, cite the source and clause/section, like "
    "(IS 2347, Clause 7) or (BIS-CERT, Section 2). If the provided context "
    "doesn't contain the answer, say so plainly instead of guessing. "
    "Always reply in the same language the user's question was written in."
)


def _format_context(chunks):
    return "\n\n".join(
        f"[{c['standard_id']} — {c['clause']}]\n{c['text']}" for c in chunks
    )


def _fallback_answer(question, chunks):
    if not chunks:
        return "No matching clause found in the indexed standards for this question."
    top = chunks[0]
    return (
        f"Based on {top['standard_id']} ({top['clause']}): {top['text']}\n\n"
        f"(No LLM key set — showing the top-matching clause directly. "
        f"Set GROQ_API_KEY or ANTHROPIC_API_KEY for a synthesized answer.)"
    )


def generate_answer(question: str, chunks: list, intent: str = None) -> str:
    context = _format_context(chunks)
    intent_line = f"(Detected query category: {intent})\n" if intent else ""
    user_msg = f"{intent_line}Question: {question}\n\nRelevant clauses:\n{context}"

    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    if anthropic_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    return _fallback_answer(question, chunks)


if __name__ == "__main__":
    from retrieval import BISRetriever
    r = BISRetriever()
    q = "What safety devices must a domestic pressure cooker have?"
    hits, intent, _ = r.search(q)
    print(generate_answer(q, hits, intent))

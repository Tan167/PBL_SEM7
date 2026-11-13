# BIS Sahayak — AI Assistant for Indian Standards & BIS Services
### (SIH26107 / SIH26108)

A hybrid **traditional NLP + LLM** conversational assistant for Indian
Standards (BIS) — answers compliance questions, recommends applicable
standards, and explains certification/hallmarking/licensing/testing
procedures, with every answer cited to its source clause.

## Why "hybrid," not just "an LLM wrapper"

Earlier version of this project ran user queries straight into TF-IDF
search and handed the result to an LLM. Feedback was that this looked
like a thin wrapper around an existing model rather than real NLP work —
fair criticism. The pipeline now looks like this:

```
User query
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ TRADITIONAL NLP LAYER (nlp_pipeline.py) — no LLM involved │
│                                                           │
│  1. Intent classification                                │
│     TF-IDF + Linear SVM, trained on a labeled query set,  │
│     predicts: standard / certification / hallmarking /    │
│     licensing / testing_lab                                │
│                                                           │
│  2. Entity/keyword extraction (spaCy)                     │
│     lemmatization, stopword removal, noun-chunking,        │
│     regex match on explicit "IS ####" standard numbers    │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ RETRIEVAL (retrieval.py)                                  │
│  - search restricted to the predicted category first       │
│  - TF-IDF cosine similarity, boosted by extracted keywords  │
│    and any explicit standard-number match                  │
│  - falls back to full-corpus search if the restricted       │
│    search finds nothing                                    │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ LLM GENERATION (generation.py) — only step touching an LLM │
│  - synthesizes a final answer from the retrieved clauses    │
│    ONLY, citing standard + clause for every claim            │
│  - offline fallback if no API key is set                    │
└─────────────────────────────────────────────────────────┘
```

The LLM never sees the raw query and never does the searching — it only
writes up an answer from a small, already-classified, already-boosted
set of clauses that classical NLP handed it. That's the concrete
difference between "call an LLM" and "combine traditional NLP and LLM,"
which was the evaluator's ask.

## How this maps to the problem statement

| Expected in PS | Where it's handled |
|---|---|
| Answer questions on Indian Standards | "Ask a compliance question" tab \u2014 intent classifier routes to `standard` category, retrieval + LLM answer |
| Recommend applicable standards for a product | "Find applicable standard" tab (`recommend_standard`) |
| Guidance on BIS certification schemes | `data/BIS_certification_scheme.txt`, its own intent category |
| Explain certification processes | `data/BIS_licensing_procedure.txt`, its own intent category |
| Consumer-related queries | Covered by the general Q&A tab (product safety, marking, etc.) |
| Hallmarking guidance | `data/BIS_hallmarking.txt`, its own intent category |
| Suggest relevant testing laboratories | `data/BIS_testing_labs.txt`, its own intent category |
| Multilingual interaction | System prompt instructs the LLM to reply in the question's language |
| Source-backed / cited responses | Every answer names standard/doc + clause/section |

## Project structure

```
bis_assistant/
├── data/                  # knowledge base: standards + BIS service guides (.txt)
├── nlp_pipeline.py        # intent classifier (TF-IDF+SVM) + spaCy keyword/entity extraction
├── ingest.py              # loads + chunks documents by Clause/Section, tags each with a category
├── retrieval.py           # intent-restricted, keyword-boosted TF-IDF search + standard recommender
├── generation.py          # LLM answer generation with citations (+ offline fallback)
├── app.py                 # Streamlit UI — shows the NLP pipeline stages, not just the final answer
└── requirements.txt
```

## Setup

```bash
cd bis_assistant
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional but recommended — without a key you still get a working demo,
# just with the raw top-matching clause instead of a synthesized answer.
export GROQ_API_KEY=your_key_here        # or:
export ANTHROPIC_API_KEY=your_key_here

streamlit run app.py
```

If `pip install` fails with an "externally managed environment" error:
`pip install -r requirements.txt --break-system-packages`, or use a venv.

## Extending the knowledge base

Drop more `.txt` files into `data/`, formatted like the existing ones,
and add the filename \u2192 category mapping in `nlp_pipeline.py`'s
`DOC_CATEGORY` dict so the intent classifier knows how to route it:

```
DOC-ID: Document Title

Clause 1: Heading
Body text...

Clause 2: Heading
Body text...
```

(`Section` works interchangeably with `Clause` \u2014 the parser accepts both.)
To improve the intent classifier itself, add more labeled examples to
`TRAINING_DATA` in `nlp_pipeline.py` \u2014 more (and more varied) examples
per category is the single highest-leverage improvement available.

## For the demo / pitch

1. Ask a product-standard question (e.g. *"What safety devices does a
   pressure cooker need?"*) \u2014 point out the "NLP pipeline" panel showing
   the detected category and extracted keywords *before* the answer, not
   just the answer itself.
2. Ask a hallmarking question (e.g. *"How do I register as a jeweller for
   hallmarking?"*) \u2014 show the classifier correctly switches category and
   searches a completely different document set.
3. Switch to "Find applicable standard", describe a product (e.g. *"I
   manufacture 14.2kg LPG cylinders"*) \u2014 show it correctly ranks IS 3196
   above unrelated standards.
4. If asked "why not just prompt an LLM to do all of this?" \u2014 answer:
   determinism and auditability. The intent classifier and keyword
   extraction are inspectable, retrainable on new data, and don't
   hallucinate; the LLM is scoped to a narrow, already-verified task
   (writing up an answer from given text), which is also why the offline
   fallback mode still works with zero API cost.

## Known limitations (fine for an internal round, worth naming if asked)

- Knowledge base is a small hand-written demo set, not the real BIS
  corpus \u2014 swapping in real standard PDFs is a data problem, not an
  architecture change (see "Extending the knowledge base" above).
- The intent classifier is trained on ~32 hand-labeled example queries \u2014
  enough to separate five fairly distinct topics, but a production
  version needs a larger, logged-query training set.
- Retrieval is TF-IDF (word-overlap based), not semantic embeddings \u2014
  works well here because clause text is fairly literal, but a synonym-
  heavy query (e.g. "kadai" instead of "pressure cooker") may miss.
  Swapping in `sentence-transformers` embeddings is a drop-in replacement
  inside `retrieval.py`'s `_score_all`.

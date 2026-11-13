"""
Retrieval layer — now driven by the traditional NLP layer (nlp_pipeline.py)
instead of raw TF-IDF-over-everything.

Pipeline per query:
  1. Classify intent (trained SVM) -> restrict search to that category's
     chunks first (e.g. a "hallmarking" question searches hallmarking
     chunks before anything else).
  2. Extract keywords/entities (spaCy lemmas + regex IS-numbers) -> boost
     the TF-IDF score of any chunk whose text contains an extracted
     keyword or an explicitly mentioned standard number.
  3. If the intent-restricted search comes up empty (rare, e.g. a vague
     query), fall back to searching the full corpus so we never return
     nothing just because classification was overconfident.

This means the LLM in generation.py only ever sees a small, already
relevant, already-labeled set of clauses — it is not doing the search.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingest import load_chunks
from nlp_pipeline import IntentClassifier, extract_keywords

KEYWORD_BOOST = 0.15
IS_NUMBER_BOOST = 0.4


class BISRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        self.intent_classifier = IntentClassifier()

        self.category_indices = {}
        for i, c in enumerate(self.chunks):
            self.category_indices.setdefault(c["category"], []).append(i)

    def _score_all(self, query: str):
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self.matrix).flatten()

    def _apply_keyword_boost(self, scores, keywords):
        lemmas = keywords["lemmas"]
        is_numbers = [n.replace(" ", "").upper() for n in keywords["is_numbers"]]
        boosted = scores.copy()
        for i, c in enumerate(self.chunks):
            text_lower = c["text"].lower()
            for lemma in lemmas:
                if lemma in text_lower:
                    boosted[i] += KEYWORD_BOOST
            chunk_std_id = c["standard_id"].replace(" ", "").upper()
            for num in is_numbers:
                if num in chunk_std_id:
                    boosted[i] += IS_NUMBER_BOOST
        return boosted

    def search(self, query: str, top_k: int = 4, use_nlp: bool = True):
        keywords = extract_keywords(query) if use_nlp else {"lemmas": [], "noun_chunks": [], "is_numbers": []}
        intent = self.intent_classifier.predict(query) if use_nlp else None

        scores = self._score_all(query)
        if use_nlp:
            scores = self._apply_keyword_boost(scores, keywords)

        candidate_indices = (
            self.category_indices.get(intent, list(range(len(self.chunks))))
            if intent else list(range(len(self.chunks)))
        )

        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)
        results = [i for i in ranked if scores[i] > 0][:top_k]

        # Fallback: if intent-restricted search found nothing, search everything
        if not results and intent:
            ranked_all = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
            results = [i for i in ranked_all if scores[i] > 0][:top_k]

        out = []
        for i in results:
            c = self.chunks[i]
            out.append({**c, "score": round(float(scores[i]), 3)})
        return out, intent, keywords

    def recommend_standard(self, product_description: str, top_k: int = 3):
        """
        Differentiator feature (ties to SIH26108): given a free-text product
        description, rank which BIS standard(s) most likely apply, at the
        document level rather than the clause level. Always searches the
        full "standard" category regardless of predicted intent, since this
        mode is specifically about matching a product to a standard.
        """
        keywords = extract_keywords(product_description)
        scores = self._score_all(product_description)
        scores = self._apply_keyword_boost(scores, keywords)

        standard_indices = self.category_indices.get("standard", [])
        seen = {}
        for i in sorted(standard_indices, key=lambda i: scores[i], reverse=True):
            if scores[i] <= 0:
                continue
            key = self.chunks[i]["standard_id"]
            if key not in seen or scores[i] > seen[key]["score"]:
                seen[key] = {**self.chunks[i], "score": round(float(scores[i]), 3)}
        ranked = sorted(seen.values(), key=lambda h: h["score"], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    r = BISRetriever()

    q = "what safety valve pressure is required for pressure cookers?"
    hits, intent, kws = r.search(q)
    print(f"Query: {q!r}")
    print(f"  detected intent: {intent}")
    for hit in hits:
        print(f"  [{hit['score']}] {hit['standard_id']} {hit['clause']} (category={hit['category']})")

    print()
    q2 = "how do I register as a jeweller for hallmarking?"
    hits2, intent2, kws2 = r.search(q2)
    print(f"Query: {q2!r}")
    print(f"  detected intent: {intent2}")
    for hit in hits2:
        print(f"  [{hit['score']}] {hit['standard_id']} {hit['clause']} (category={hit['category']})")

    print("\nRecommend standard for: 'I manufacture LPG gas cylinders for home kitchens'")
    for hit in r.recommend_standard("I manufacture LPG gas cylinders for home kitchens"):
        print(f"  [{hit['score']}] {hit['standard_id']}")

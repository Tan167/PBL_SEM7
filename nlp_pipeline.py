"""
Traditional NLP layer — runs BEFORE retrieval and BEFORE the LLM sees
anything. This is the piece that was missing: the original version treated
retrieval as a search step and left everything else to the LLM. Here:

  1. Intent classification: a trained TF-IDF + Linear SVM classifier
     (classic bag-of-words ML, not an LLM call) predicts which BIS service
     category a query belongs to — standard, certification, hallmarking,
     licensing, or testing_lab. This narrows retrieval to the right subset
     of the knowledge base instead of searching everything.

  2. Entity/keyword extraction: spaCy does lemmatization + noun-chunk
     extraction (standard NLP preprocessing), plus a regex pass for
     explicit "IS ####" standard-number mentions. Extracted keywords are
     used to boost retrieval scores for chunks that mention them.

Only the FINAL answer synthesis step (generation.py) touches an LLM. Intent
routing and keyword extraction are both classical NLP, trained/run locally.
"""

import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

_nlp = spacy.load("en_core_web_sm")

IS_NUMBER_PATTERN = re.compile(r"\bIS[-\s]?\d{2,5}\b", re.IGNORECASE)

# Small labeled training set for the intent classifier. In a production
# system this would be built from real logged queries; for the prototype
# it's hand-labeled example queries per category, enough for a linear
# bag-of-words classifier to separate five fairly distinct topics.
TRAINING_DATA = [
    # standard (product safety/spec questions)
    ("what safety devices does a pressure cooker need", "standard"),
    ("maximum temperature rise allowed for electrical appliances", "standard"),
    ("tds limit for packaged drinking water", "standard"),
    ("small parts test for toys under 3 years", "standard"),
    ("wall thickness requirement for lpg cylinders", "standard"),
    ("what is the safety valve pressure for cookers", "standard"),
    ("marking requirements for household appliances", "standard"),
    ("flammability test for toy materials", "standard"),
    ("hydrostatic test frequency for gas cylinders", "standard"),
    ("ph value allowed in drinking water", "standard"),
    # certification
    ("how do I get an ISI mark license for my product", "certification"),
    ("what is the compulsory registration scheme CRS", "certification"),
    ("difference between voluntary and mandatory certification", "certification"),
    ("how much does BIS certification cost", "certification"),
    ("how long is a BIS certification licence valid", "certification"),
    ("steps to apply for product certification", "certification"),
    ("what documents are needed for ISI certification", "certification"),
    # hallmarking
    ("what is a HUID number", "hallmarking"),
    ("how do I register as a jeweller for hallmarking", "hallmarking"),
    ("purity grades for gold jewellery hallmarking", "hallmarking"),
    ("is hallmarking mandatory for gold jewellery", "hallmarking"),
    ("how to verify hallmark on the BIS Care app", "hallmarking"),
    ("silver hallmarking standard", "hallmarking"),
    # licensing
    ("what is the process for getting a BIS manufacturing licence", "licensing"),
    ("how to renew a BIS licence", "licensing"),
    ("what happens during a factory inspection for licensing", "licensing"),
    ("documents required to apply for a licence on manak online", "licensing"),
    ("how long does licence renewal take", "licensing"),
    # testing_lab
    ("which lab can test my product against a BIS standard", "testing_lab"),
    ("where is the nearest BIS recognized laboratory", "testing_lab"),
    ("does BIS have its own testing labs", "testing_lab"),
    ("how to find a NABL accredited lab for testing", "testing_lab"),
    ("list of BIS labs in my region", "testing_lab"),
]

DOC_CATEGORY = {
    "IS_302_household_electrical_appliances.txt": "standard",
    "IS_14543_packaged_drinking_water.txt": "standard",
    "IS_9873_toy_safety.txt": "standard",
    "IS_2347_pressure_cookers.txt": "standard",
    "IS_3196_lpg_cylinders.txt": "standard",
    "BIS_certification_scheme.txt": "certification",
    "BIS_hallmarking.txt": "hallmarking",
    "BIS_licensing_procedure.txt": "licensing",
    "BIS_testing_labs.txt": "testing_lab",
}


class IntentClassifier:
    def __init__(self):
        texts = [t for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        self.clf = LinearSVC()
        self.clf.fit(X, labels)

    def predict(self, query: str) -> str:
        X = self.vectorizer.transform([query])
        return self.clf.predict(X)[0]

    def predict_with_confidence(self, query: str):
        X = self.vectorizer.transform([query])
        label = self.clf.predict(X)[0]
        # LinearSVC has no predict_proba by default; use decision_function
        # margin as a rough confidence signal instead of a fabricated number.
        margins = self.clf.decision_function(X)[0]
        top_margin = max(margins) if hasattr(margins, "__iter__") else margins
        return label, float(top_margin)


def extract_keywords(query: str):
    """Classical NLP preprocessing: lemmatize, drop stopwords/punctuation,
    pull noun chunks, and regex-match explicit IS standard numbers."""
    doc = _nlp(query)
    lemmas = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and t.is_alpha]
    noun_chunks = [c.text.lower() for c in doc.noun_chunks]
    is_numbers = [m.group(0).upper().replace(" ", " ") for m in IS_NUMBER_PATTERN.finditer(query)]
    return {
        "lemmas": lemmas,
        "noun_chunks": noun_chunks,
        "is_numbers": is_numbers,
    }


if __name__ == "__main__":
    clf = IntentClassifier()
    tests = [
        "What safety devices does a pressure cooker need?",
        "How do I register as a jeweller for hallmarking?",
        "Which lab can test my product against IS 2347?",
        "How much does an ISI licence renewal cost?",
    ]
    for q in tests:
        label, margin = clf.predict_with_confidence(q)
        kws = extract_keywords(q)
        print(f"Q: {q}")
        print(f"  intent={label} (margin={margin:.2f})")
        print(f"  keywords={kws}")
        print()

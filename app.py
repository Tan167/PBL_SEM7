import streamlit as st

from retrieval import BISRetriever
from generation import generate_answer

st.set_page_config(page_title="BIS Sahayak", page_icon="📋", layout="centered")

st.title("📋 BIS Sahayak — AI Assistant for Indian Standards & BIS Services")
st.caption(
    "Hybrid pipeline: classical NLP (intent classification + entity "
    "extraction) narrows and boosts retrieval, then an LLM only synthesizes "
    "the final cited answer — it never does the searching itself."
)


@st.cache_resource
def get_retriever():
    return BISRetriever()


retriever = get_retriever()

tab_ask, tab_recommend = st.tabs(["🔎 Ask a compliance question", "🏷️ Find applicable standard"])

with tab_ask:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the maximum TDS allowed in packaged drinking water?",
    )
    if st.button("Get answer", type="primary") and question:
        with st.spinner("Classifying intent and extracting keywords..."):
            hits, intent, keywords = retriever.search(question)

        with st.expander("🧠 NLP pipeline (what ran before the LLM)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Detected category (SVM classifier):**  \n`{intent}`")
            with c2:
                kw_display = keywords["lemmas"][:8]
                st.markdown(f"**Extracted keywords (spaCy lemmas):**  \n`{', '.join(kw_display) if kw_display else '—'}`")
            if keywords["is_numbers"]:
                st.markdown(f"**Standard numbers mentioned:** `{', '.join(keywords['is_numbers'])}`")
            st.caption("This intent + these keywords narrowed and boosted the retrieval search below — the LLM only sees what's left.")

        if not hits:
            st.warning("No matching clause found in the indexed standards.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, hits, intent)
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            for h in hits:
                with st.expander(f"{h['standard_id']} — {h['clause']}  (relevance {h['score']}, category={h['category']})"):
                    st.text(h["text"])

with tab_recommend:
    st.write("Describe your product and see which BIS standard(s) likely apply.")
    product = st.text_area(
        "Product description",
        placeholder="e.g. I manufacture 14.2 kg LPG cylinders for household kitchens",
    )
    if st.button("Find standard") and product:
        with st.spinner("Matching against indexed standards..."):
            recs = retriever.recommend_standard(product)
        if not recs:
            st.warning("No matching standard found.")
        else:
            for rec in recs:
                st.markdown(f"**{rec['standard_id']}** — relevance {rec['score']}")
                st.caption(rec["text"][:200] + "...")

st.divider()
st.caption(
    "Demo knowledge base: 5 sample product standards (household appliances, "
    "packaged drinking water, toys, pressure cookers, LPG cylinders) plus "
    "certification scheme, hallmarking, licensing, and testing-lab guides. "
    "Intent classifier trained on a small hand-labeled query set — "
    "swap in real logged queries to improve it further."
)

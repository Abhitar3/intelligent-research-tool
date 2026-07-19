import streamlit as st
from rag import (
    process_urls,
    generate_answer,
    generate_research_brief,
    compare_papers,
    extract_methods_and_results,
)

def display_sources(sources):
    if not sources:
        return

    st.subheader("Evidence Snippets")

    for source in sources:
        st.markdown(f"**Source {source['source_number']}**: {source['source']}")
        st.write(source["snippet"])


st.title("Research Paper Assistant")

url1 = st.sidebar.text_input("Paper URL 1")
url2 = st.sidebar.text_input("Paper URL 2")
url3 = st.sidebar.text_input("Paper URL 3")

process_url_button = st.sidebar.button("Process Papers")

placeholder = st.empty()

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != ""]
    if len(urls) == 0:
        placeholder.text("You must provide at least one paper URL.")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

brief_tab, compare_tab, methods_tab, question_tab = st.tabs(
    ["Research Brief", "Compare Papers", "Methods & Results", "Ask Question"]
)

with brief_tab:
    brief_button = st.button("Generate Research Brief")

    if brief_button:
        try:
            brief, sources = generate_research_brief()
            st.header("Research Brief")
            st.write(brief)
            display_sources(sources)
        except RuntimeError:
            placeholder.text("You must process papers first.")

with compare_tab:
    comparison_goal = st.text_input(
        "Comparison goal",
        placeholder="Example: Which method is best for small datasets?",
    )

    compare_button = st.button("Compare Papers")

    if compare_button:
        try:
            comparison, sources = compare_papers(comparison_goal)
            st.header("Paper Comparison")
            st.write(comparison)
            display_sources(sources)
        except RuntimeError:
            placeholder.text("You must process papers first.")

with methods_tab:
    methods_button = st.button("Extract Methods And Results")

    if methods_button:
        try:
            methods_results, sources = extract_methods_and_results()
            st.header("Methods And Results")
            st.write(methods_results)
            display_sources(sources)
        except RuntimeError:
            placeholder.text("You must process papers first.")

with question_tab:
    query = st.text_input("Ask a question about the papers")

    if query:
        try:
            answer, sources = generate_answer(query)
            st.header("Answer")
            st.write(answer)
            display_sources(sources)
        except RuntimeError:
            placeholder.text("You must process papers first.")
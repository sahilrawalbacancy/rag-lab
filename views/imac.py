"""Page 3 - short, visual document (Apple iMac Product Environmental Report)."""

import pandas as pd
import streamlit as st

import ui


def explorer(doc_id: str) -> None:
    st.markdown("#### Text layer vs vision, page by page")
    ui.vision_compare(doc_id)

    st.markdown("#### Footnote links")
    st.caption("Claims cite endnotes pages away. Each link below is followed at answer time, "
               "so the claim and its conditions reach the model together.")
    index = ui.load_index(f"{doc_id}__advanced")
    st.dataframe(pd.DataFrame([{
        "claim (page)": d.metadata["page"], "section": d.metadata["section"][:40],
        "cites": d.metadata["footnotes"],
        "claim": d.page_content[:120].replace("\n", " "),
    } for d in index.docs if d.metadata.get("footnotes")]), hide_index=True, width="stretch")


ui.assignment_page("imac", "3-imac.md", explorer)

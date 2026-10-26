"""Page 2 - long, structured report (World Bank Global Economic Prospects)."""

import pandas as pd
import streamlit as st

import ui
from rag import structured


def explorer(doc_id: str) -> None:
    st.markdown("#### The section map the index is built on")
    rows: dict = {}
    for d in ui.load_index(f"{doc_id}__advanced").docs:
        m = d.metadata
        row = rows.setdefault((m["chapter"], m["section"]), {
            "chapter": m["chapter"], "section": "" if m["section"] == m["chapter"] else m["section"],
            "pages": set(), "chunks": 0, "table chunks": 0, "from vision": 0,
        })
        row["pages"].add(m["page"])
        row["chunks"] += 1
        row["table chunks"] += m["kind"] == "table"
        row["from vision"] += m["origin"] == "vision"
    table = [{**r, "pages": f"{min(r['pages'])}-{max(r['pages'])}"} for r in rows.values()]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    st.markdown("#### Try the router")
    q = st.text_input("A question", "What is the growth outlook for South Asia in this report?",
                      key="gep_route_q")
    where, label = structured.route(q)
    if where:
        st.success(f"Searched only inside **{label}** - filter `{where}`")
    else:
        st.info("No section named - the whole report is searched.")

    st.markdown("#### Table pages: text layer vs vision")
    ui.vision_compare(doc_id)


ui.assignment_page("gep", "2-world-bank.md", explorer)

"""
RAG Lab - one Streamlit app: the initial chatbot plus one page per assignment.

    .venv/bin/python -m streamlit run streamlit_app.py

Streamlit Community Cloud runs this file by default. Pages live in views/
(not pages/: st.navigation ignores that folder by design).
"""

import streamlit as st

st.set_page_config(page_title="RAG Lab", page_icon="🧪", layout="wide")

st.navigation({
    "": [st.Page("views/overview.py", title="Overview", icon="🏠", default=True)],
    "Initial chatbot": [
        st.Page("views/compliance_copilot.py", title="Compliance Copilot", icon="📋"),
    ],
    "Advanced RAG": [
        st.Page("views/gpo.py", title="1 · GPO Annual Report", icon="📄"),
        st.Page("views/world_bank.py", title="2 · World Bank GEP", icon="📚"),
        st.Page("views/imac.py", title="3 · Apple iMac Report", icon="📊"),
    ],
}).run()

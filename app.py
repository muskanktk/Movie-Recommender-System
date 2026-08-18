import streamlit as st


def Pages():
    welcomePg = st.Page("pages/welcome.py",
     title="Dashboard", 
     icon=":material/home:" 
    )

    mainPg = st.Page("pages/recommender.py",
    title="Movie Recommender Generator",
    icon=":material/chat:"
    )

    analysispg = st.Page("pages/analysis.py",
    title="Analytical Page",
    icon="📊"
    )

    Navigation= st.navigation([welcomePg, mainPg, analysispg])

    Navigation.run()

if __name__ == "__main__":
    Pages()

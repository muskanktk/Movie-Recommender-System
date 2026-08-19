import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Analytics",
    layout="wide"
)

st.title("Analytics")

tableau_url = (
    "https://public.tableau.com/views/"
    "Movies_17871083157580/Dashboard1"
    "?:showVizHome=no"
)

components.iframe(
    tableau_url,
    height=1400,
    scrolling=True
)
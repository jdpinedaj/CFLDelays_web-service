# poetry run streamlit run regression_app.py

import streamlit as st
import pandas as pd
import os
from utils import (
    make_predictions,
    analysis_db_llm,
    MODEL,
)

os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"


def main():
    st.set_page_config(
        page_title="CFL - Rail intermodal operation disruptions",
        page_icon=":train:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.image(image="images/cfl-logo.png")

    # Selecting which analysis to show using buttons
    analysis = st.sidebar.radio(
        "Select analysis",
        [
            "Predictions",
            "Explorations_LLM",
        ],
    )

    if analysis == "Predictions":
        make_predictions(MODEL)
    elif analysis == "Explorations_LLM":
        analysis_db_llm()
    else:
        st.write("Please select an analysis")


if __name__ == "__main__":
    main()

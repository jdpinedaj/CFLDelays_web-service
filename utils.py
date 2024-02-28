# Model setup # Using Pycaret

from pycaret.regression import predict_model, load_model
import streamlit as st
import requests
import json
import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Tuple, Any
from text import (
    database_schema,
    instructions,
    information_tables,
    some_examples,
)

#! Load model
MODEL = load_model("./model/model")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]
RELAY_URL = st.secrets["RELAY_URL"]


PG_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

#!######################################
#!######################################
#!#####  SUB FUNCTIONS #################
#!######################################
#!######################################


def _predict_delay(model: object, df: object) -> str:
    """
    This function takes the model and the input features as input and returns the prediction.
    Args:
        model: trained model
        df: input features
    Returns:
        str: prediction
    """
    predictions_data = predict_model(estimator=model, data=df)
    if predictions_data["Label"][0] >= 0:
        return "late.", predictions_data["Label"][0]
    elif predictions_data["Label"][0] < 0:
        return "early.", predictions_data["Label"][0] * (-1)


def _generate_sql_from_natural_language(
    question: str, model: str = "gpt-4-turbo-preview"
) -> str:
    """
    Generate SQL query from a natural language question using the OpenAI model.
    Args:
        question (str): The natural language question to generate SQL from.
        model (str): The model to use for generating the SQL query, defaulting to "gpt-4-turbo-preview".
    Returns:
        str: The generated SQL query.
    """
    api_key = OPENAI_API_KEY
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.1,  # Adjust temperature as needed
    }

    try:
        response = requests.post(
            RELAY_URL,
            json=data,
            headers=headers,
        )
        response_data = response.json()
        sql_query = response_data["choices"][0]["message"]["content"].strip()

        # Post-process the SQL query as needed
        sql_query = sql_query.strip(".")
        sql_query = sql_query.replace('"', "'")

        return sql_query

    except Exception as e:
        st.error(f"Error occurred while generating the SQL query: {e}")
        return "There was an error generating the SQL query."


def _execute_query(response_text: str) -> Tuple[List[Tuple[Any, ...]], List[str]]:
    """
    This function executes the SQL query found in the response text and returns the result.
    Args:
        response_text (str): The response text containing the SQL query.
    Returns:
        Tuple[List[Tuple[Any, ...]], List[str]]: The result of the SQL query as a list of rows (each row is a tuple of values),
        and the columns of the SQL query result as a list of strings.
    """
    # Create a SQLAlchemy engine
    engine = create_engine(PG_URI)

    # Pattern to match SQL queries starting with SELECT or WITH, case-sensitive, and ending with a semicolon
    query_pattern = r"(SELECT|WITH)\s.*?;"

    # Search for SQL query within the response text
    match = re.search(query_pattern, response_text, re.DOTALL)

    if not match:
        st.error("No SQL query found in the response text.")
        return []

    # Extract the SQL query from the match
    sql_query = match.group(0)

    try:
        with engine.connect() as connection:
            # Execute the SQL query
            result = connection.execute(text(sql_query))
            # Fetch all results
            rows = result.fetchall()
            columns = result.keys()
            return rows, columns
    except SQLAlchemyError as e:
        st.error(f"Error occurred while executing the SQL query: {e}")
        return [], []


def _serialize_sql_result(row: tuple) -> dict:
    """
    Serialize an individual row of an SQL query result to a dictionary.
    Args:
        row (tuple): A single row of the SQL query result.
    Returns:
        dict: The serialized row as a dictionary.
    """
    serialized_row = {}
    for i, value in enumerate(row):
        # Convert complex types (like UUID) to their string representation
        if hasattr(value, "hex"):  # Check if the value is a UUID
            serialized_value = str(value)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            serialized_value = str(value)  # Fallback to string representation
        else:
            serialized_value = value
        serialized_row[f"column_{i}"] = serialized_value
    return serialized_row


def _transform_sql_result_into_natural_language(
    first_request: str, result: list, model: str = "gpt-4-turbo-preview"
) -> str:
    """
    Transforms the result of an SQL query into natural language using the OpenAI model via a relay.
    Args:
        first_request (str): The original request that generated the SQL query.
        result (list): The result of the SQL query.
        model (str): The model to use for the transformation, defaulting to "gpt-4-turbo-preview".
    Returns:
        str: The result of the transformation in natural language.
    """
    api_key = OPENAI_API_KEY
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Convert result to a list of dictionaries if not already in that format
    # This handles cases where result might contain non-serializable objects like SQLAlchemy Row objects
    serialized_result = [_serialize_sql_result(row) for row in result]
    result_json = json.dumps(serialized_result)

    # Create the prompt
    prompt = f"Original request: {first_request}\nSQL query result: {result_json}\nExplain the outcome in natural language."

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Adjust temperature as needed
    }

    try:
        response = requests.post(
            RELAY_URL,
            json=data,
            headers=headers,
        )
        response_data = response.json()
        transformed_result = response_data["choices"][0]["message"]["content"].strip()

        return transformed_result
    except Exception as e:
        st.error(f"Error occurred while transforming the SQL query result: {e}")
        return "There was an error transforming the SQL query result."


def _reset_later_steps() -> None:
    """
    Reset the session state for steps 2, 3, and 4.
    Args:
        None
    Returns:
        None
    """
    for key in ["sql_query_for_edit", "modified_sql_query", "query_result"]:
        if key in st.session_state:
            del st.session_state[key]


#!######################################
#!######################################
#!#####  FUNCTIONS #####################
#!######################################
#!######################################


def make_predictions(model: object) -> None:
    """
    This function is used to make predictions based on the input features.
    Args:
        model: trained model
    Returns:
        None
    """
    st.header("Prediction of rail intermodal operation disruptions")
    st.write(
        "This is a web app to predict the arrival delay time of the trains (in minutes)."
    )
    st.write(
        "Please adjust the value of each feature in the sidebar, and then click on the Predict button at the bottom to see the prediction of the model."
    )

    #! Input features
    teu_count = st.sidebar.slider(
        label="TEU count", min_value=0, max_value=100, value=60, step=1
    )

    train_length = st.sidebar.slider(
        label="Train length [m]", min_value=100, max_value=720, value=550, step=10
    )

    train_weight = st.sidebar.slider(
        label="Train weight [t]", min_value=290, max_value=2300, value=1200, step=10
    )

    wagon_count = st.sidebar.slider(
        label="Number of wagons", min_value=6, max_value=35, value=15, step=1
    )

    total_distance_trip = st.sidebar.slider(
        label="Total distance of the trip [km]",
        min_value=80,
        max_value=1450,
        value=650,
        step=10,
    )

    departure_delay = st.sidebar.slider(
        label="Departure delay [min]",
        min_value=-250,
        max_value=950,
        value=30,
        step=5,
    )

    distance_between_control_stations = st.sidebar.slider(
        label="Distance between control stations [km]",
        min_value=1,
        max_value=800,
        value=70,
        step=5,
    )

    #! Converting input features to a dataframe

    features = {
        "teu_count": teu_count,
        "train_length": train_length,
        "train_weight": train_weight,
        "wagon_count": wagon_count,
        "total_distance_trip": total_distance_trip,
        "departure_delay": departure_delay,
        "distance_between_control_stations": distance_between_control_stations,
    }

    features_df = pd.DataFrame([features])
    features_df["weight_per_length_of_train"] = round(
        features_df["train_weight"] / features_df["train_length"], 1
    )
    features_df["weight_per_wagon_of_train"] = round(
        features_df["train_weight"] / features_df["wagon_count"], 1
    )
    features_df.drop(columns=["train_weight", "wagon_count"], axis=1, inplace=True)

    ##!Display of results
    st.table(
        features_df.rename(
            columns={
                "teu_count": "TEU count",
                "train_length": "Train length [m]",
                "total_distance_trip": "Total distance of the trip [km]",
                "departure_delay": "Departure delay [min]",
                "distance_between_control_stations": "Distance between control stations [km]",
                "weight_per_length_of_train": "Weight per length [t/m]",
                "weight_per_wagon_of_train": "Weight per wagon [t/wagon]",
            }
        )
    )

    if st.button("Predict"):
        prediction = _predict_delay(model, features_df)  # Using pycaret
        st.header(
            " Based on feature values, the trip will arrive "
            + str(round(prediction[1], 1))
            + " minutes "
            + str(prediction[0])
        )  #   Using pycaret


def analysis_db_llm():
    """ """
    # use columns to organize layout
    col1, col2 = st.columns(2)

    with col1:
        # Step 1: User input for natural language query and generate SQL and its explanation
        st.subheader("Step 1: Generate SQL Query and Explanation")
        # Creating a layout with columns inside col1 for finer control over the text_input width
        col1_text_input, col1_2 = st.columns([4, 1])

        with col1_text_input:
            user_query = st.text_input(
                "Enter your query in natural language:",
                key="user_query",
            )

        if (
            "user_query_last" in st.session_state
            and st.session_state["user_query_last"] != user_query
        ):
            # Call reset_later_steps if the user query has changed
            _reset_later_steps()

        st.session_state["user_query_last"] = user_query

        generate_sql_button = st.button(
            "Generate SQL Query and Explanation", key="gen_sql"
        )

        if generate_sql_button and user_query:
            question = database_schema + user_query
            full_output = _generate_sql_from_natural_language(question)
            sql_query = (
                full_output.split("```sql")[1].split("```")[0].strip()
            )  # Extract the SQL query
            st.session_state["full_output"] = full_output
            st.session_state["sql_query_for_edit"] = sql_query
            # Display the full output from Step 1
            st.markdown(full_output)
        elif "full_output" in st.session_state:
            # Always display the full output from Step 1 if it exists, regardless of any action in Step 2
            st.markdown(st.session_state["full_output"])

        # Step 2: Modify the generated SQL query
        st.subheader("Step 2: Modify the SQL query as needed")
        if "sql_query_for_edit" in st.session_state:
            modified_sql_query = st.text_area(
                "SQL Query:", value=st.session_state["sql_query_for_edit"], height=300
            )
            set_modified_sql_button = st.button("Set Modified SQL Query")
            if set_modified_sql_button:
                st.session_state["modified_sql_query"] = modified_sql_query
                st.success("SQL query updated. Proceed to execute the query.")

    with col2:
        # Step 3: Execute the SQL query and display results
        st.subheader("Step 3: Execute SQL Query and Display Results")
        execute_sql_button = st.button("Execute SQL Query", key="exec_sql")
        if execute_sql_button and "modified_sql_query" in st.session_state:
            rows, columns = _execute_query(st.session_state["modified_sql_query"])
            if rows:
                # Convert query results into a pandas DataFrame
                df = pd.DataFrame(rows, columns=columns)
                st.session_state["query_result"] = rows
                st.session_state["query_result_df"] = df
            else:
                st.error(
                    "No results found or an error occurred during query execution."
                )

        if "query_result_df" in st.session_state:
            st.dataframe(st.session_state["query_result_df"])

        # Step 4: Transform Query Result into Natural Language
        st.subheader("Step 4: Transform Result into Natural Language")
        transform_result_button = st.button("Transform Result", key="trans_res")
        if (
            transform_result_button
            and "query_result" in st.session_state
            and st.session_state["query_result"]
        ):
            transformed_result = _transform_sql_result_into_natural_language(
                st.session_state.get("user_query", ""), st.session_state["query_result"]
            )
            if transformed_result:
                st.write(transformed_result)
            else:
                st.error(
                    "No transformation result found or an error occurred during transformation."
                )

    st.markdown("---")

    bottom_col1, bottom_col2 = st.columns([1, 1])

    with bottom_col1:
        #  Showing the instructions
        st.subheader("Instructions")
        st.markdown(instructions)

        #  Showing some examples
        st.subheader("Some Examples")
        st.markdown(some_examples)

    with bottom_col2:
        #  Showing the information tables
        st.subheader("Information Tables")
        st.markdown(information_tables)
        pass

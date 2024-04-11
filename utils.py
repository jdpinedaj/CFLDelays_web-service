# Model setup # Using Pycaret

from pycaret.regression import predict_model, load_model
import streamlit as st
import requests
import json
import re
from datetime import datetime, date
import pandas as pd
import base64
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


# OpenAI parameters
OPENAI_MODEL = "gpt-4-0125-preview"  # "gpt-4-turbo-preview"  # "gpt-4-0125-preview" # For more options, see: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
TEMPERATURE_GENERATE_SQL = 0.1
TOP_P_GENERATE_SQL = 0.1
TEMPERATURE_TRANSFORM_RESULT = 0.1
TOP_P_TRANSFORM_RESULT = 0.1
MAX_TOKENS_TRANSFORM_RESULT = 100
#  The previous values are based on https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
SHOW_USER_QUERY = True
SHOW_FULL_OUTPUT = False
SHOW_TABLE = True
SHOW_TRANSFORMED_RESULT = True

# Create a SQLAlchemy engine
engine = create_engine(PG_URI)

CONTEXT = """
Database Overview:
- The database features the 'public_etl.df_final_etl_no_outliers' table.

TABLE AND DATA TYPES:
Table: 'public_etl.df_final_etl_no_outliers':
- Columns: incoterm(VARCHAR), max_teu(FLOAT), teu_count(FLOAT), max_length(FLOAT), train_length(FLOAT), train_weight(FLOAT), 
    planned_departure_day(VARCHAR), planned_arrival_day(VARCHAR), departure_week_number(FLOAT), wagon_count(FLOAT), total_distance_trip(FLOAT), 
    sum_tares_wagons(FLOAT), departure_country(VARCHAR), arrival_country(VARCHAR), departure_delay(FLOAT), arrival_delay(FLOAT), 
    distance_between_control_stations(FLOAT), weight_per_length_of_train(FLOAT), weight_per_wagon_of_train(FLOAT), incident_type(VARCHAR), 
    incident_gravity(VARCHAR), incident_customer_reason(VARCHAR), month_arrival(VARCHAR), arrival_night(VARCHAR), peak_time(VARCHAR)

DESCRIPTION:
    incoterm: The incoterm of the trip.
    max_teu: The maximum number of TEUs (Twenty-foot Equivalent Units) transported in the trip.
    teu_count: The total number of TEUs transported in the trip.
    max_length: The maximum length of the train.
    train_length: The length of the train [m].
    train_weight: The weight of the train [t].
    planned_departure_day: The planned day of departure of the trip.
    planned_arrival_day: The planned day of arrival of the trip.
    departure_week_number: The week number of the planned departure day.
    wagon_count: The number of wagons in the train [wagons].
    total_distance_trip: The total distance of the trip [km].
    sum_tares_wagons: The sum of the tares of the wagons [t].
    departure_country: The country of departure.
    arrival_country: The country of arrival.
    departure_delay: The delay of the departure [min].
    arrival_delay: The delay of the arrival [min].
    distance_between_control_stations: The distance between control stations [km].
    weight_per_length_of_train: The weight per length of the train [t/m].
    weight_per_wagon_of_train: The weight per wagon of the train [t/wagon].
    incident_type: The type of incident.
    incident_gravity: The gravity of the incident.
    incident_customer_reason: The reason of the incident.
    month_arrival: The month of the arrival.
    arrival_night: If the train arrived at night [yes/no].
    peak_time: If the train arrived at peak time [yes/no].

FIRST ROWS:
    incoterm,max_teu,teu_count,max_length,train_length,train_weight,planned_departure_day,planned_arrival_day,departure_week_number,wagon_count,total_distance_trip,sum_tares_wagons,departure_country,arrival_country,departure_delay,arrival_delay,distance_between_control_stations,weight_per_length_of_train,weight_per_wagon_of_train,incident_type,incident_gravity,incident_customer_reason,month_arrival,arrival_night,peak_time
    -1,43.5,42.75,502,322,747.093,Monday   ,Monday   ,50,10,120.64814681172649,324700,Luxembourg,Belgium,-13,-13,53.02470713863994,2.3201645962732917,74.7093,no_incident,no_incident,no_incident,December ,no,no
    EXW,90,78.12,700,606,1045.773,Monday   ,Tuesday  ,7,30,73.26746752917097,570980,Luxembourg,Belgium,1471,1460,53.02470713863994,1.72569801980198,34.8591,no_incident,no_incident,no_incident,February ,no,yes
    -1,67.5,41.53,502,478,1049.735,Wednesday,Wednesday,9,15,53.02489700923587,463020,Luxembourg,Belgium,-37,-33,53.02470713863994,2.1960983263598326,69.98233333333333,no_incident,no_incident,no_incident,March    ,no,no
    -1,67.5,48.75,502,482,1100.494,Monday   ,Monday   ,11,15,53.02489700923587,471420,Luxembourg,Belgium,13,21,53.02470713863994,2.2831825726141077,73.36626666666666,no_incident,no_incident,no_incident,March    ,no,no
    -1,67.5,42.25,502,478,1071.33,Monday   ,Monday   ,33,15,73.26746752917097,464980,Luxembourg,Belgium,4,-9,53.02470713863994,2.241276150627615,71.422,no_incident,no_incident,no_incident,August   ,no,no
\n\n
"""

HIGHLIGHTS = """
Query Construction Guidelines:
- Generate one SQL query per request.
- Adhere strictly to PostgreSQL syntax, with emphasis on efficient use of data types and partitioning.
- PLEASE ONLY USE THE VARIABLES AND TABLES MENTIONED IN THE DESCRIPTION, and only give me the query to get the data.
- Keep in mind to add explicits casts depending on the data types.
- Please organize the structure of the SQL queries in a way that is easy to read and understand.

Additional Considerations:
- Ensure queries are logically structured and devoid of in-line comments.
- Use previous queries and the provided context as a guide for structuring new requests.

\n\n
"""

#!######################################
#!######################################
#!#####  SUB FUNCTIONS #################
#!######################################
#!######################################


#! Sub functions for predictions


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


#! Sub functions for analysis_db_llm


def _analysis_generate_sql_from_natural_language(
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


def _analysis_execute_query(
    response_text: str,
) -> Tuple[List[Tuple[Any, ...]], List[str]]:
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


def _analysis_serialize_sql_result(row: tuple) -> dict:
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


def _analysis_transform_sql_result_into_natural_language(
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
    serialized_result = [_analysis_serialize_sql_result(row) for row in result]
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


def _analysis_reset_later_steps() -> None:
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


#! Sub functions for chat_db_llm


def _initialize_session_states():
    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state["sql_query_results"] = None
    # Initialize a unique identifier for the user_query input to reset it
    if "reset_counter" not in st.session_state:
        st.session_state["reset_counter"] = 0
    #  Storing the time of the query
    if "query_time" not in st.session_state:
        st.session_state["query_time"] = datetime.now()


def _convert_datetime(obj: Any) -> Any:
    """
    Convert datetime objects to ISO format for JSON serialization.
    Args:
        obj (Any): The object to serialize.
    Returns:
        Any: The serialized object.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(
        "Object of type '{}' is not serializable".format(type(obj).__name__)
    )


def _download_data_csv():
    """
    Creates a download button in the Streamlit app to download the latest SQL query result as a CSV file.
    """
    # Check if the query results are available in the session state
    if (
        "sql_query_results" in st.session_state
        and st.session_state["sql_query_results"]
    ):
        rows, columns = st.session_state["sql_query_results"]

        # Convert the query results into a Pandas DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()

        # Create a link for downloading
        href = f'<a href="data:file/csv;base64,{b64_csv}" download="query_results.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        # Display a message if there's no data to download
        st.write("No data available to download.")


def _clear_chat_and_restart_button():
    # Button to clear chat history and reset the app state
    if st.button("Clear Chat and Restart"):
        st.session_state["chat_history"] = []
        # Increment reset_counter to reset the user_query text input
        st.session_state["reset_counter"] += 1
        # Reset other states as needed
        st.rerun()


def _generate_sql_from_natural_language(
    question: str,
    model: str,
    temperature: float,
    top_p: float,
) -> str:
    """
    Generate SQL query from a natural language question using the OpenAI model.
    Args:
        question (str): The natural language question to generate SQL from.
        model (str): The model to use for generating the SQL query.
        temperature (float): The temperature to use for generating the SQL query. Sets the sampling temperature between 0 and 2.
        top_p (float): The top_p value to use for generating the SQL query. Uses nucleus sampling; considers tokens with top_p probability mass.
    Returns:
        str: The generated SQL query.
    For more information about parameters, see: https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models#2-an-example-chat-completion-api-call
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,  # Adjust temperature as needed
        # "max_tokens": MAX_TOKENS,
        "top_p": top_p,
        # "frequency_penalty": FREQUENCY_PENALTY,
        # "presence_penalty": PRESENCE_PENALTY,
    }

    try:
        response = requests.post(
            RELAY_URL,
            json=data,
            headers=headers,
        )
        response_data = response.json()
        full_output = response_data["choices"][0]["message"]["content"].strip()

        # Post-process the SQL query as needed
        full_output = full_output.strip(".")
        full_output = full_output.replace('"', "'")

        return full_output

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
    # Trim and normalize query to catch hidden or embedded non-select/with statements
    query_trimmed = response_text.strip()
    query_normalized = re.sub(
        r"\s+", " ", query_trimmed
    ).upper()  # Replace multiple spaces with a single space and uppercase

    # Check if the query starts with "SELECT" or "WITH"
    if not query_normalized.startswith("SELECT") and not query_normalized.startswith(
        "WITH"
    ):
        error_msg = "Invalid query: Only SELECT or WITH statements are allowed."
        st.error(error_msg)
        return [], []

    # Ensure the query ends with a semicolon for security
    if not query_normalized.endswith(";"):
        error_msg = "The SQL query must end with a semicolon."
        st.error(error_msg)
        return [], []

    # Ensure no DML or DDL operations are embedded
    dml_ddl_keywords = [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "ALTER ",
        "DROP ",
        "CREATE ",
        "TRUNCATE ",
    ]
    if any(keyword in query_normalized for keyword in dml_ddl_keywords):
        error_msg = "Security alert: Modification operations are not permitted."
        st.error(error_msg)
        return [], []

    # # Pattern to match SQL queries starting with SELECT or WITH, case-sensitive, and ending with a semicolon
    # query_pattern = r"(SELECT|WITH)\s.*?;"

    # # Search for SQL query within the response text
    # match = re.search(query_pattern, response_text, re.DOTALL)

    # if not match:
    #     st.error("No SQL query found in the response text.")
    #     return []

    # # Extract the SQL query from the match
    # sql_query = match.group(0)

    try:
        with engine.connect() as connection:
            # Execute the SQL query
            result = connection.execute(
                text(query_trimmed)
            )  # Execute trimmed original query
            # result = connection.execute(text(sql_query))
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
    first_request: str,
    chat_context: str,
    rows: list,
    columns: list,
    model: str,
    temperature: float,
    top_p: float,
) -> str:
    """
    Transforms the result of an SQL query into natural language using the OpenAI model via a relay.
    Args:
        first_request (str): The original request that generated the SQL query.
        chat_context (str): The chat context to provide additional information for the transformation.
        rows (list): The result of the SQL query as a list of rows (each row is a tuple of values).
        columns (list): The columns of the SQL query result as a list of strings.
        model (str): The model to use for the transformation.
        temperature (float): The temperature to use for the transformation.
        top_p (float): The top_p value to use for the transformation.
    Returns:
        str: The result of the transformation in natural language.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    # Convert result to a list of dictionaries if not already in that format
    # This handles cases where result might contain non-serializable objects like SQLAlchemy Row objects
    serialized_rows = [dict(zip(columns, row)) for row in rows]
    result_json = json.dumps(
        serialized_rows, indent=2, default=_convert_datetime
    )  # Pretty print the JSON for readability

    # Creating the prompt with columns and rows included
    columns_formatted = ", ".join(columns)  # Formatting column names for readability
    # prompt = f"Original request: {first_request} \SQL query result: {result_json}\nExplain the outcome in natural language."
    prompt = (
        f"Original request: {first_request}\n\n"
        f"Chat context: {chat_context}\n\n"
        f"Columns: {columns_formatted}\n\n"
        f"SQL query result (JSON):\n{result_json}\n\n"
        "Explain the outcome in natural language, answering in terms of the original request."
    )

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,  # Adjust temperature as needed
        # "max_tokens": MAX_TOKENS,
        "top_p": top_p,
        # "frequency_penalty": FREQUENCY_PENALTY,
        # "presence_penalty": PRESENCE_PENALTY,
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
        return "There was an error transforming the SQL query result. Please better describe the information you need."


def _execute_all_in_chat(user_query: str) -> None:
    """
    Handles the entire process of generating an SQL query from a natural language question,
    executing the query, and transforming the result back into natural language, updating the
    chat history accordingly.
    Args:
        user_query (str): The user's natural language query.
    Returns:
        None
    """
    # ? Check for export/save/table in the user query
    # export_request_keywords = ["export", "save", "data in csv", "create a table"]
    # is_export_request = any(
    #     keyword in user_query.lower() for keyword in export_request_keywords
    # )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Extract chat history texts for SQL context
    chat_history_texts = [msg["content"] for msg in st.session_state["chat_history"]]
    chat_history_combined = "\n".join(chat_history_texts)

    # Generate the SQL query from the natural language question
    question = f"{CONTEXT}\n\n---\nCHAT HISTORY:\n{chat_history_combined}\n---\nUSER REQUEST:\n{user_query}\n\n{HIGHLIGHTS}"
    try:
        full_output = _generate_sql_from_natural_language(
            question,
            OPENAI_MODEL,
            TEMPERATURE_GENERATE_SQL,
            TOP_P_GENERATE_SQL,
        )
        if SHOW_FULL_OUTPUT:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"SQL Query: {full_output}"}
            )

        # Check if an SQL query was generated
        if "```sql" not in full_output:
            raise ValueError("Failed to generate SQL query.")

        sql_query = full_output.split("```sql")[1].split("```")[0].strip()

        # Execute the SQL query
        rows, columns = _execute_query(sql_query)

        # After successfully executing the query and obtaining rows and columns
        st.session_state["sql_query_results"] = (rows, columns)

        # Transform SQL query result into natural language
        transformed_result = _transform_sql_result_into_natural_language(
            user_query,
            chat_history_combined,
            rows,
            columns,
            OPENAI_MODEL,
            TEMPERATURE_TRANSFORM_RESULT,
            TOP_P_TRANSFORM_RESULT,
        )

        # Update chat history with the SQL query and its natural language transformation
        if SHOW_USER_QUERY:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"SQL Query: {sql_query}"}
            )
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": transformed_result}
        )

        # Update chat history combined
        chat_history_combined = "\n".join(
            [msg["content"] for msg in st.session_state["chat_history"]]
        )

    except Exception as e:
        error_message = f"An error occurred: {str(e)} {full_output}"
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": error_message}
        )
        st.error(error_message)

        # Update chat history combined
        chat_history_combined = "\n".join(
            [msg["content"] for msg in st.session_state["chat_history"]]
        )

    # Rerun the app to reflect updates in the UI
    st.rerun()


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
            _analysis_reset_later_steps()

        st.session_state["user_query_last"] = user_query

        generate_sql_button = st.button(
            "Generate SQL Query and Explanation", key="gen_sql"
        )

        if generate_sql_button and user_query:
            question = database_schema + user_query
            full_output = _analysis_generate_sql_from_natural_language(question)
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
            rows, columns = _analysis_execute_query(
                st.session_state["modified_sql_query"]
            )
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
            transformed_result = _analysis_transform_sql_result_into_natural_language(
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


def chat_db_llm():

    # Ensure all session states are initialized
    _initialize_session_states()

    # Display previous chat messages
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for natural language query
    user_input = st.chat_input("Enter your query:", key="user_query")

    if user_input:
        # Add user input to chat history to display it
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # Process the user input
        _execute_all_in_chat(user_input)

    # Creating a spacer and the restart button at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([0.2, 0.4, 0.2])

    with col1:
        _download_data_csv()

    with col3:
        _clear_chat_and_restart_button()

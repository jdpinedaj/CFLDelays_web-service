database_schema = """
POSTGRESQL DATABASE SCHEMA NAME: 'public_etl'
######
- Table 1 name: 'public_etl.df_final_etl_no_outliers'.
    COLUMN STRUCTURE: incoterm(VARCHAR), max_teu(FLOAT), teu_count(FLOAT), max_length(FLOAT), train_length(FLOAT), train_weight(FLOAT), 
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

    first rows:
    incoterm,max_teu,teu_count,max_length,train_length,train_weight,planned_departure_day,planned_arrival_day,departure_week_number,wagon_count,total_distance_trip,sum_tares_wagons,departure_country,arrival_country,departure_delay,arrival_delay,distance_between_control_stations,weight_per_length_of_train,weight_per_wagon_of_train,incident_type,incident_gravity,incident_customer_reason,month_arrival,arrival_night,peak_time
    -1,43.5,42.75,502,322,747.093,Monday   ,Monday   ,50,10,120.64814681172649,324700,Luxembourg,Belgium,-13,-13,53.02470713863994,2.3201645962732917,74.7093,no_incident,no_incident,no_incident,December ,no,no
    EXW,90,78.12,700,606,1045.773,Monday   ,Tuesday  ,7,30,73.26746752917097,570980,Luxembourg,Belgium,1471,1460,53.02470713863994,1.72569801980198,34.8591,no_incident,no_incident,no_incident,February ,no,yes
    -1,67.5,41.53,502,478,1049.735,Wednesday,Wednesday,9,15,53.02489700923587,463020,Luxembourg,Belgium,-37,-33,53.02470713863994,2.1960983263598326,69.98233333333333,no_incident,no_incident,no_incident,March    ,no,no
    -1,67.5,48.75,502,482,1100.494,Monday   ,Monday   ,11,15,53.02489700923587,471420,Luxembourg,Belgium,13,21,53.02470713863994,2.2831825726141077,73.36626666666666,no_incident,no_incident,no_incident,March    ,no,no
    -1,67.5,42.25,502,478,1071.33,Monday   ,Monday   ,33,15,73.26746752917097,464980,Luxembourg,Belgium,4,-9,53.02470713863994,2.241276150627615,71.422,no_incident,no_incident,no_incident,August   ,no,no

######
    HIGHLIGHTS:
    - PLEASE ONLY USE THE VARIABLES AND TABLES MENTIONED IN THE DESCRIPTION, and only give me the query to get the data.
    - Keep in mind that the SQL queries should be written in PostgreSQL syntax.
    - Keep in mind to add explicits casts depending on the data types.
    - Please organize the structure of the SQL queries in a way that is easy to read and understand.
\n
"""

instructions = """
- **Step 1:** Generate SQL Query from Natural Language: Enter a natural language question and click the "Generate SQL Query and Explanation" button to generate an SQL query from the question.
- **Step 2:** Modify the Generated SQL Query: Modify the generated SQL query as needed and click the "Set Modified SQL Query" button to proceed.
- **Step 3:** Execute the SQL Query and Display Results: Click the "Execute SQL Query" button to execute the SQL query and display the results.
- **Step 4:** Transform Query Result into Natural Language: Click the "Transform Result" button to transform the result of the SQL query into natural language.
"""

information_tables = """
- Table 1: `public_etl.df_final_etl_no_outliers`  
    - incoterm(VARCHAR)
    - max_teu(FLOAT)
    - teu_count(FLOAT)
    - max_length(FLOAT)
    - train_length(FLOAT)
    - train_weight(FLOAT)
    - planned_departure_day(VARCHAR)
    - planned_arrival_day(VARCHAR)
    - departure_week_number(FLOAT)
    - wagon_count(FLOAT)
    - total_distance_trip(FLOAT)
    - sum_tares_wagons(FLOAT)
    - departure_country(VARCHAR)
    - arrival_country(VARCHAR)
    - departure_delay(FLOAT)
    - arrival_delay(FLOAT)
    - distance_between_control_stations(FLOAT)
    - weight_per_length_of_train(FLOAT)
    - weight_per_wagon_of_train(FLOAT)
    - incident_type(VARCHAR)
    - incident_gravity(VARCHAR)
    - incident_customer_reason(VARCHAR)
    - month_arrival(VARCHAR)
    - arrival_night(VARCHAR)
    - peak_time(VARCHAR)
"""

some_examples = """
- Give me the number of trains that arrived at peak time in February
- Give me the number of stations with more incidents in August
- Give me the number of trains with a delay in the departure in December
- Give me the number of trains that passed through Belgium and France in January
- How many trains had a weight per length of train greater than 2.5 in February?
"""

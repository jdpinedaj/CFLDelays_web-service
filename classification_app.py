import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


base_dir = './'

st.image(image='images/cfl-logo.png')
st.title('Short-term predictive model')
st.header('Prediction of rail intermodal operation disruptions') 
st.write('This is a web app to classify if a train will be on time (< 1 hour), delayed (1-3 hours) or too delayed (> 3 hours).')
st.write('Please adjust the value of each feature in the sidebar, and then click on the Predict button at the bottom to see the prediction of the classifier.')


# Model setup # Using Pycaret
def predict_delay(model, df):    
    predictions_data = predict_model(estimator = model, data = df)
    if predictions_data['Label'][0]=='0_on_time':
        return 'on time (less than 1 hour).', str(round(predictions_data['Score'][0]*100,1))+'%'
    elif predictions_data['Label'][0]=='1_late':
        return 'delayed (between 1 and 3 hours).', str(round(predictions_data['Score'][0]*100,1))+'%'
    elif predictions_data['Label'][0]=='2_too_late':
        return 'too delayed (more than 3 hours).', str(round(predictions_data['Score'][0]*100,1))+'%'
  
  
# Model setup
#def predict_delay_pickle(df):    
#    predictions_data = model_pickle.predict(df)
#    if str(predictions_data)=='[0]':
#        return 'on time (less than 1 hour)'
#    elif str(predictions_data)=='[1]':
#        return 'delayed (between 1 and 3 hours)'
#    elif str(predictions_data)=='[2]':
#        return 'too delayed (more than 3 hours)'
    

    
    
model = load_model(base_dir + 'lightgbm_short_term')                           # Using Pycaret
#model_pickle = pd.read_pickle(base_dir + 'model.pmml')


# Input features

teu_count = st.sidebar.slider(label = 'TEU count', min_value = 0,
                              max_value = 100,
                              value = 60,
                              step = 1)

train_length = st.sidebar.slider(label = 'Train length [m]', min_value = 100,
                                 max_value = 720,
                                 value = 550,
                                 step = 10)
                          
train_weight = st.sidebar.slider(label = 'Train weight [t]', min_value = 290,
                                 max_value = 2300,
                                 value = 1200,
                                 step = 10)                          

wagon_count = st.sidebar.slider(label = 'Number of wagons', min_value = 6,
                                max_value = 35,
                                value = 15,
                                step = 1)

total_distance_trip = st.sidebar.slider(label = 'Total distance of the trip [km]', min_value = 80,
                                        max_value = 1450,
                                        value = 650,
                                        step = 10)


departure_delay = st.sidebar.slider(label = 'Departure delay [min]', min_value = -250,
                                    max_value = 950,
                                    value = 30,
                                    step = 5)
   

distance_between_control_stations = st.sidebar.slider(label = 'Distance between control stations [km]', min_value = 1,
                                                      max_value = 800,
                                                      value = 70,
                                                      step = 5)


# Converting input features to a dataframe

features = {'teu_count': teu_count, 'train_length': train_length,
            'train_weight': train_weight, 'wagon_count': wagon_count,
            'total_distance_trip': total_distance_trip, 'departure_delay': departure_delay,
            'distance_between_control_stations': distance_between_control_stations
            }
 

features_df  = pd.DataFrame([features])
features_df['weight_per_length_of_train'] = round(features_df['train_weight'] / features_df['train_length'], 1)
features_df['weight_per_wagon_of_train'] = round(features_df['train_weight'] / features_df['wagon_count'], 1)
features_df.drop(columns=['train_weight', 'wagon_count'], axis=1, inplace=True)


## Display of results
st.table(features_df.rename(columns=
                            {'teu_count': 'TEU count', 'train_length': 'Train length [m]', 'total_distance_trip': 'Total distance of the trip [km]',
                             'departure_delay': 'Departure delay [min]', 'distance_between_control_stations': 'Distance between control stations [km]',
                             'weight_per_length_of_train': 'Weight per length [t/m]', 'weight_per_wagon_of_train': 'Weight per wagon [t/wagon]'}
                            ))  

if st.button('Predict'):
    prediction = predict_delay(model, features_df)          # Using pycaret
    #prediction_pickle =  predict_delay_pickle(features_df)
    st.header(' Based on feature values, there is a probability of ' + str(prediction[1]) + ' that the trip will be '+ str(prediction[0]))      #   Using pycaret
    


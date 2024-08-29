import pandas as pd
import streamlit as st
from darts.timeseries import TimeSeries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from darts.dataprocessing.transformers import Scaler
from darts.models import RandomForest



session = st.session_state

actual = pd.read_excel('./data/filtered_actuals.xlsx')

target = pd.read_excel('./data/filtered_broker_target_filledna.xlsx')

confidence_score = pd.read_excel('./data/confidence_score.xlsx')

actual_series = TimeSeries.from_dataframe(actual, time_col='FiscalYear')

broker_target_series = TimeSeries.from_dataframe(target, time_col='FiscalYear')


brokers = actual.columns[1:]

st.set_page_config(page_title="Jill Brokerage Target",page_icon="📈",layout="wide")
st.markdown("#### Forecasted Broker Target")


with st.sidebar:
    input_broker = st.selectbox("Choose Broker",brokers, placeholder='Haggar, James')
    input_broker_target = st.number_input(label='Input broker\'s target')
    session['input_borker'] = input_broker
    session['input_broker_target'] = input_broker_target

actual_ts = actual_series[input_broker]

actual_scaler = Scaler()

scaled_target_series = actual_scaler.fit_transform(actual_ts)
name = scaled_target_series.columns[0]

rf = RandomForest(lags=1, n_estimators=10)
rf.fit(scaled_target_series)
prediction_rf = rf.predict(1)

scaled_predicted = actual_scaler.inverse_transform(prediction_rf)

scaled_predicted_df = scaled_predicted.pd_dataframe().reset_index()

actual_ts_df = actual_ts.pd_dataframe().tail(1).reset_index()


session['actual_ts_df'] = actual_ts_df

predicted_df = pd.concat([actual_ts_df,scaled_predicted_df])

session['predicted_df'] = predicted_df

st.json(session)














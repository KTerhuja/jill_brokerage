import pandas as pd
import streamlit as st
from darts.timeseries import TimeSeries
from src import train, utils
import matplotlib.pyplot as plt
import plotly.express as px

#load data

## actual data
session = st.session_state

actual_target_data = pd.read_excel('./data/filtered_filledna.xlsx')

actual_target_data_ts = TimeSeries.from_dataframe(actual_target_data, time_col="FiscalYear")

external_factors = pd.read_excel('./data/full_dataset_multivariate_gdp_interstRates.xlsx')

interest_rates_series = TimeSeries.from_series(external_factors[external_factors['Broker']=='Weatherby, Samuel'].set_index('FiscalYear')['Interest Rate'])

gdp_series = TimeSeries.from_series(external_factors[external_factors['Broker']=='Weatherby, Samuel'].set_index('FiscalYear')['Texas GDP'])

brokers = actual_target_data.columns[1:]
types_external_factors = ['None','Interest Rates', 'GDP']

st.set_page_config(page_title="Jill Brokerage Target",page_icon="📈",layout="wide")
st.markdown("#### Forecasted Broker Target")

with st.sidebar:
    input_broker = st.selectbox("Choose Broker",brokers)
    input_external_factor = st.selectbox("External Variables", types_external_factors)
    session['input_borker'] = input_broker
    session['input_external_factor'] = input_external_factor


l1,r1 = st.columns([2,2])

input_years = l1.select_slider("Forecast Upto Year",[2024+i for i in range(3)])
session['input_years'] = input_years


    
target_train_ts = actual_target_data_ts[input_broker]

predicted = train.train_ts(target_train_ts,interest_rates_series,gdp_series,session['input_years'], session['input_external_factor'])


session['predicted'] = predicted


plot_df = utils.combine(target_train_ts,predicted)
# st.dataframe(plot_df)

# st.dataframe(plot_df)

l,r = st.columns([3,1])

with l:
    fig = px.line(
        plot_df,
        x=plot_df.index,
        y=f"{input_broker}",
        color="tag",
        title=f"Forecasted Broker target",
        color_discrete_sequence=["mediumspringgreen","crimson"],
        height=600
        )
    
    fig.update_layout({ 'plot_bgcolor': '#4E6E81'})
    st.plotly_chart(fig)

with r: 

    predicted = predicted.pd_dataframe()
    st.dataframe(predicted)
    
#     fig.update_layout(
#     xaxis_title="Year", yaxis_title="New HIV Population"
# )









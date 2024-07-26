import pandas as pd
import streamlit as st
from darts.timeseries import TimeSeries
from src import train, utils
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

#load data

## actual data

cutoff_year = '2021'
session = st.session_state

actual_target_data = pd.read_excel('./data/filtered_filledna.xlsx')

actual_target_data_ts = TimeSeries.from_dataframe(actual_target_data, time_col="FiscalYear")
train_series, val_series = actual_target_data_ts.split_before(pd.Timestamp(cutoff_year))

external_factors = pd.read_excel('./data/full_dataset_multivariate_gdp_interstRates.xlsx')

interest_rates_series = TimeSeries.from_series(external_factors[external_factors['Broker']=='Weatherby, Samuel'].set_index('FiscalYear')['Interest Rate'])

interest_rates_series, _ = interest_rates_series.split_before(pd.Timestamp(cutoff_year))

gdp_series = TimeSeries.from_series(external_factors[external_factors['Broker']=='Weatherby, Samuel'].set_index('FiscalYear')['Texas GDP'])
gdp_series, _ = gdp_series.split_before(pd.Timestamp(cutoff_year))

brokers = train_series.columns[1:]
types_external_factors = ['None','Interest Rates', 'GDP']

st.set_page_config(page_title="Jill Brokerage Target",page_icon="📈",layout="wide")
st.markdown("#### Forecasted Broker Target")


with st.sidebar:
    input_broker = st.selectbox("Choose Broker",brokers, placeholder='Haggar, James')
    input_external_factor = st.selectbox("External Variables", types_external_factors)
    session['input_borker'] = input_broker
    session['input_external_factor'] = input_external_factor
    


l1,r1 = st.columns([0.5,2])

# input_years = l1.select_slider("Forecast Upto Year",[2021+i for i in range(3)])
input_years = 2023
session['input_years'] = input_years


    
target_train_ts = train_series[input_broker]
ground_truth_ts = val_series[input_broker]


predicted = train.train_ts(target_train_ts,interest_rates_series,gdp_series,session['input_years'], session['input_external_factor'])



predicted_df = predicted.pd_dataframe().reset_index()

target_train_df = target_train_ts.pd_dataframe().tail(1).reset_index()

target_train_df['lower'] = target_train_df[input_broker]
target_train_df['upper'] = target_train_df[input_broker]


session['target_train_df'] = target_train_df

predicted_df = pd.concat([target_train_df,predicted_df])



true_df = val_series

session['predicted'] = predicted_df

ground_truth_df = ground_truth_ts.pd_dataframe().reset_index()
target_train_df_tail = target_train_df.tail(1)

ground_truth_df = pd.concat([target_train_df_tail,ground_truth_df ])

session['ground_truth'] = ground_truth_df


l, r = st.columns([2.7,1])

with l:

    st.markdown('')

    plot_df = utils.combine(target_train_ts,predicted).rename(columns = {input_broker: 'Broker Target ($)'})

    fig_line = px.line(
            plot_df,
            x=plot_df.index,
            y='Broker Target ($)',
            color="tag",
            title=f"Forecasted Broker target",
            color_discrete_sequence=["green","crimson"],
            height=700,
            width= 1000
            )


    fig_ground_truth = go.Figure()
    fig_ground_truth.add_trace(go.Scatter(x=ground_truth_df['FiscalYear'], y=ground_truth_df[input_broker], mode='lines', line_color = 'blue', name='Ground Truth'))
    fig_area = go.Figure()

    fig_area.add_trace(go.Scatter(x=predicted_df['FiscalYear'], y =predicted_df['upper'],
        fill=None,
        mode='lines',
        line_color='indigo',
        showlegend = False
        ))
    fig_area.add_trace(go.Scatter(
        x=predicted_df['FiscalYear'], y =predicted_df['lower'],
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='indigo',
        showlegend = False
        ))



    fig_combined = go.Figure(data=fig_line.data + fig_area.data + fig_ground_truth.data)

    # Update layout for combined figure
    fig_combined.update_layout(
        title=f'Forecasted target for {input_broker}',
        title_x=0.3,
        xaxis_title='Year',
        yaxis_title='Value',
        height=700,
        width=1000
    )

    # Show the combined figure
    # fig_combined.show()  
    fig_combined.update_layout({ 'plot_bgcolor': '#F5EDED'})
    st.plotly_chart(fig_combined)
    # st.plotly_chart(fig_area)
    # st.dataframe(plot_df)

    # st.dataframe(plot_df)


session['predicted'] = predicted.pd_dataframe()
with r:
    st.markdown(f'✦ Predicted Target range for {input_broker} ')
    predicted = predicted.pd_dataframe().rename(columns = {input_broker: 'Predicted Target ($)'})

    for cols in predicted.select_dtypes(include=['float64']):
        predicted[cols] = predicted[cols].astype('int')
    
 
    style_predicted = utils.style_df(predicted)
    st.table(style_predicted)


# st.json(session)

st.caption("""
<style>body
{zoom: 70%;}
</style>
""",
unsafe_allow_html
=True) 

# st.json(session)









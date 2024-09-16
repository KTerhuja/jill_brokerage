import pandas as pd
import streamlit as st
from darts.timeseries import TimeSeries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from darts.dataprocessing.transformers import Scaler
from darts.models import RandomForest
from src import utils
from darts.models import (
    RNNModel,
    BlockRNNModel)



session = st.session_state

cutoff_year = '2023'

actual = pd.read_excel('./data/filtered_actuals.xlsx')

target = pd.read_excel('./data/filtered_broker_target_filledna.xlsx')

diffs = pd.read_excel('./data/deficit_values.xlsx')

actual['FiscalYear'] = pd.to_datetime(actual['FiscalYear'], format = "%Y")
diffs['FiscalYear'] = pd.to_datetime(diffs['FiscalYear'], format = "%Y")

confidence_score = pd.read_excel('./data/confidence_score.xlsx')

actual_series = TimeSeries.from_dataframe(actual, time_col='FiscalYear')

broker_target_series = TimeSeries.from_dataframe(target, time_col='FiscalYear')

diffs_series = TimeSeries.from_dataframe(diffs, time_col='FiscalYear')


diffs.set_index('FiscalYear',inplace=True)
diffs['Avg Shortfall'] = diffs.apply(lambda x: x.mean(), axis=1)

individual_shortfall = diffs.apply(lambda x: x.mean())

train_series, val_series = actual_series.split_before(pd.Timestamp(cutoff_year))

diffs_train_series, diffs_val_series = diffs_series.split_before(pd.Timestamp(cutoff_year))


brokers = actual.columns[1:]

st.set_page_config(page_title="Jill Brokerage Target",page_icon="📈",layout="wide")
# st.markdown("#### Forecasted Broker Target")


with st.sidebar:
    st.image('./data/jll_background.png')
    input_broker = st.selectbox("Choose Broker",brokers, index=6)
    input_broker_target = st.number_input(label='Input broker\'s target')
    session['input_borker'] = input_broker
    session['input_broker_target'] = input_broker_target

actual_ts = actual_series[input_broker]

train_actual_ts = train_series[input_broker]

val_actual_ts = val_series[input_broker]

diffs_train_ts = diffs_train_series[input_broker]

diffs_val_ts = diffs_val_series[input_broker]

extended_subset_series = diffs_train_ts.append_values([input_broker_target])

train_actual_df = train_actual_ts.pd_dataframe().reset_index()


session['train_actual_df'] = train_actual_df


val_actual_df = val_actual_ts.pd_dataframe().reset_index()

# predicted_df = pd.concat([train_actual_df, prediction_df])

val_plot_df = pd.concat([train_actual_df.tail(1), val_actual_df])

session['val_plot_df'] = val_plot_df




if input_broker_target:

    #---------------training----------------------
    # actual_scaler = Scaler()

    # # scaled_target_series = actual_scaler.fit_transform(actual_ts)

    actual_scaler, diff_scaler = Scaler(), Scaler()

    # scaled_actual_ts = actual_scaler.fit_transform(actual_ts)

    scaled_actual_train_ts = actual_scaler.fit_transform(train_actual_ts)

    scaled_actual_val_ts = actual_scaler.transform(val_actual_ts)
    
    scaled_diff_ts = diff_scaler.fit_transform(diffs_train_ts)

    scaled_extended_ts = diff_scaler.transform(extended_subset_series)

    rf = RandomForest(lags=1, n_estimators=10, lags_future_covariates=[0])

    rf.fit(scaled_actual_train_ts, future_covariates = scaled_diff_ts)

    prediction_rf = rf.predict(1, future_covariates=scaled_extended_ts)

    inverse_prediction = actual_scaler.inverse_transform(prediction_rf)

    #---------------training----------------------

    predicted_value = inverse_prediction[0].pd_series().values[0]
    val_value = val_actual_ts[0].pd_series().values[0]

    deficit_value = input_broker_target - predicted_value

    diffs_extend_ts = diffs_train_ts.append_values([deficit_value])

    diffs_extend_df = diffs_extend_ts.pd_dataframe().reset_index()

    scaled_actual_train_ts = scaled_actual_train_ts.pd_dataframe().reset_index()

    prediction_df = inverse_prediction.pd_dataframe().reset_index()

    session['scaled_actual_train_ts'] = scaled_actual_train_ts

    predicted_df = pd.concat([train_actual_df, prediction_df])

    session['predicted_df'] = predicted_df


    meta_data = pd.DataFrame({'Year': ['2024'],'Actual': [int(val_value)], 'Predicted': [int(predicted_value)]})

    #-------------plot-------------------------

    tab1, tab2 = st.tabs(['Predicted Target', 'Individual Shorfall'])

    with tab1:
    
        left_plot, right_plot = st.columns([2,1])

        with left_plot:

            plot_df = utils.combine(train_actual_df, prediction_df)

            plot_df['FiscalYear'] = pd.to_datetime(plot_df['FiscalYear'], format = '%Y')

            session['plot_data'] = plot_df

            fig_line = px.line(
                plot_df,
                x=plot_df['FiscalYear'],
                y=f'{input_broker}',
                color="tag",
                title=f"Forecasted Broker target",
                color_discrete_sequence=["green","crimson"],
                height=500,
                width= 1000
                )

            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=val_plot_df['FiscalYear'], y=val_plot_df[input_broker], mode='lines', line_color = 'yellow', name='Ground Truth'))

            
            fig_broker_target = go.Figure()
            fig_broker_target.add_trace(go.Scatter(x=diffs_extend_df['FiscalYear'], y=diffs_extend_df[input_broker], mode='lines', line_color = 'blue', name='Shortfall'))



            fig_combined = go.Figure(data=fig_line.data + fig_val.data + fig_broker_target.data )

            # Update layout for combined figure
            fig_combined.update_layout(
                title=f'Forecasted Sales for {input_broker}',
                title_x=0.3,
                xaxis_title='Year',
                yaxis_title='Value',
                height=450,
                width=1000
            )
            st.plotly_chart(fig_combined)

            st.dataframe(meta_data,hide_index= True)

            




        with right_plot:

            shortfall_yoy = go.Figure()
            shortfall_yoy.add_trace(go.Scatter(x=diffs.index, y=diffs['Avg Shortfall'], mode='lines', line_color = 'blue', name='Average Shortfall YoY'))

            shortfall_yoy.update_layout(
                title=f'Average Shortfall YoY',
                title_x=0.3,
                xaxis_title='Year',
                yaxis_title='Shorfall',
                height=450,
                width=1000
            )
            st.plotly_chart(shortfall_yoy)



    with tab2:

        individual_shortfall = individual_shortfall.drop('Avg Shortfall')

        individual_shortfall_yoy = go.Figure()
        individual_shortfall_yoy.add_trace(go.Bar(
            x=individual_shortfall.index, 
            y=individual_shortfall.values, 
            marker_color='blue', 
            name='Average Individual Shortfall'
        ))

        st.plotly_chart(individual_shortfall_yoy)






        

# st.json(session)

# st.caption("""
# <style>body
# {zoom: 80%;}
# </style>
# """,
# unsafe_allow_html
# =True) 
















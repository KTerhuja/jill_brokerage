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

actual = pd.read_excel('./data/filtered_actuals.xlsx')

target = pd.read_excel('./data/filtered_broker_target_filledna.xlsx')

diffs = pd.read_excel('./data/adjusted_values.xlsx')

confidence_score = pd.read_excel('./data/confidence_score.xlsx')

actual_series = TimeSeries.from_dataframe(actual, time_col='FiscalYear')

broker_target_series = TimeSeries.from_dataframe(target, time_col='FiscalYear')

diffs_series = TimeSeries.from_dataframe(diffs, time_col='FiscalYear')


brokers = actual.columns[1:]

st.set_page_config(page_title="Jill Brokerage Target",page_icon="📈",layout="wide")
# st.markdown("#### Forecasted Broker Target")


with st.sidebar:
    input_broker = st.selectbox("Choose Broker",brokers, placeholder='Haggar, James')
    input_broker_target = st.number_input(label='Input broker\'s target')
    session['input_borker'] = input_broker
    session['input_broker_target'] = input_broker_target

actual_ts = actual_series[input_broker]

broker_target_ts = broker_target_series[input_broker].pd_dataframe().reset_index()

diffs_subset = diffs[['FiscalYear', input_broker]]

extended = pd.DataFrame({'FiscalYear':[2024], input_broker:[input_broker_target]})

extended_subset = pd.concat([diffs_subset, extended])

extended_subset_series = TimeSeries.from_dataframe(extended_subset, time_col='FiscalYear')

if input_broker_target:

    #---------------training----------------------
    # actual_scaler = Scaler()

    # # scaled_target_series = actual_scaler.fit_transform(actual_ts)
    

    # rf = RandomForest(lags=1, n_estimators=10)
    # rf.fit(actual_series[input_broker], future_covariates=diffs_series[input_broker])
    # prediction_rf = rf.predict(1, future_covariates=extended_subset_series)


    # scaled_predicted_df = prediction_rf.pd_dataframe().reset_index()

   
    model_name = "RNN_test"
    model_futcov = RNNModel(
        model="LSTM",
        hidden_dim=6,
        batch_size=2,
        n_epochs=10,
        random_state=0,
        training_length=3,
        input_chunk_length=2,
        model_name=model_name,
        save_checkpoints=True,  # store model states: latest and best performing of validation set
        force_reset=True
    )

    model_futcov.fit(
    series= actual_ts[input_broker],
    future_covariates= diffs_series[input_broker]
)
    
    predicted = model_futcov.predict(1, future_covariates=extended_subset_series)

    #---------------training----------------------

    st.write(predicted)

    # actual_ts_df = actual_ts.pd_dataframe().reset_index()

    # session['actual_ts_df'] = actual_ts_df

    # predicted_df = pd.concat([actual_ts_df, scaled_predicted_df])

    # session['predicted_df'] = predicted_df

    # plot_df = utils.combine(actual_ts_df, scaled_predicted_df)

    # session['plot_data'] = plot_df

    # plot_df['FiscalYear'] = pd.to_datetime(plot_df['FiscalYear'], format = '%Y')

    # session['plot_data'] = plot_df



    # # session['plot_data'] = plot_df

    # plot_df['FiscalYear'] = pd.to_datetime(plot_df['FiscalYear'], format = '%Y')

    # fig_line = px.line(
    #     plot_df,
    #     x=plot_df['FiscalYear'],
    #     y=f'{input_broker}',
    #     color="tag",
    #     title=f"Forecasted Broker target",
    #     color_discrete_sequence=["green","crimson"],
    #     height=700,
    #     width= 1000
    #     )



    # fig_broker_target = go.Figure()
    # fig_broker_target.add_trace(go.Scatter(x=diffs_subset['FiscalYear'], y=diffs_subset[input_broker], mode='lines', line_color = 'blue', name='Broker Target'))



    # fig_combined = go.Figure(data=fig_line.data + fig_broker_target.data)

    # # Update layout for combined figure
    # fig_combined.update_layout(
    #     title=f'Forecasted target for {input_broker}',
    #     title_x=0.3,
    #     xaxis_title='Year',
    #     yaxis_title='Value',
    #     height=700,
    #     width=1000
    # )

    # # Show the combined figure
    # # fig_combined.show()  
    # # fig_combined.update_layout({ 'plot_bgcolor': '#F5EDED'})
    # st.plotly_chart(fig_combined)

# st.json(session)

st.caption("""
<style>body
{zoom: 80%;}
</style>
""",
unsafe_allow_html
=True) 
















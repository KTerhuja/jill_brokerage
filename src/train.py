import pandas as pd
import streamlit as st
import pandas as pd
from darts import TimeSeries
from darts.metrics.metrics import rmse
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel, NBEATSModel, RandomForest, NHiTSModel



def train_ts(target_train_ts,interest_rates_series,gdp_series,input_years, input_external_factor):

    target_scaler, ir_scaler, gdp_scaler = Scaler(), Scaler(), Scaler()

    scaled_target_series = target_scaler.fit_transform(target_train_ts)
    scaled_ir_series = ir_scaler.fit_transform(interest_rates_series)
    scaled_gdp_series = gdp_scaler.fit_transform(gdp_series)

    if input_external_factor =='None':
        rf = RandomForest(lags=1, n_estimators=10)
        rf.fit(scaled_target_series)
        prediction_rf = rf.predict(input_years - 2023)

        scaled_predicted = target_scaler.inverse_transform(prediction_rf)

        return scaled_predicted

    elif input_external_factor == 'Interest Rates':

        rf = RandomForest(lags=1, n_estimators=10)
        rf.fit([scaled_target_series,scaled_ir_series])
        prediction_rf = rf.predict(input_years - 2023,scaled_target_series)

        scaled_predicted = target_scaler.inverse_transform(prediction_rf)
        
        return scaled_predicted
    
    elif input_external_factor == 'GDP':
        rf = RandomForest(lags=1, n_estimators=10)
        rf.fit([scaled_target_series,scaled_gdp_series])
        prediction_rf = rf.predict(input_years - 2023,scaled_target_series)

        scaled_predicted = target_scaler.inverse_transform(prediction_rf)
        
        return scaled_predicted
    


import pandas as pd
from darts.timeseries import TimeSeries

def combine(target_train, predicted):

    target_df = target_train.pd_dataframe()
    target_df['tag'] = 'target'
    

    predicted_df = predicted.pd_dataframe()
    connector = target_df.tail(1)
    connector['tag'] = 'predicted'
    
    predicted_df['tag'] = 'predicted'

    concat = pd.concat([target_df,connector, predicted_df] )

    return concat



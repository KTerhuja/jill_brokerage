import pandas as pd
from darts.timeseries import TimeSeries

def combine(target_train, predicted):

    # target_df = target_train.pd_dataframe()
    target_train['tag'] = 'target'
    

    # predicted_df = predicted.pd_dataframe()
    connector = target_train.tail(1)
    connector['tag'] = 'predicted'
    
    predicted['tag'] = 'predicted'

    concat = pd.concat([target_train,connector, predicted] )

    return concat


# Function to apply color to the 'Predicted Target ($)' column
def color_predicted(val):
    return 'color: red'

# Function to apply color to the 'lower' and 'upper' columns
def color_lower_upper(val):
    return 'color: green'


def style_df(df):

    styled_df = df.style.applymap(color_predicted, subset=['Predicted Target ($)'])\
                    .applymap(color_lower_upper, subset=['lower', 'upper'])\
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#2C2E3E')]}
                    ])

    
    return styled_df


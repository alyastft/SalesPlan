import pandas as pd

def preprocess_data(df):

    month_mapping = {
        'Mei': 'May',
        'Agu': 'Aug',
        'Okt': 'Oct',
        'Des': 'Dec'
    }

    df['Date'] = df['Date'].astype(str)

    for indo, eng in month_mapping.items():

        df['Date'] = df['Date'].str.replace(indo, eng)

    df['Date'] = pd.to_datetime(
        df['Date']
    )

    df = df.rename(columns={
        'Date': 'ds',
        'Sales': 'y'
    })

    return df

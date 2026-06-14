import pandas as pd


def preprocess_data(df):

    month_mapping = {
        'Jan': 'Jan',
        'Feb': 'Feb',
        'Mar': 'Mar',
        'Apr': 'Apr',
        'Mei': 'May',
        'Jun': 'Jun',
        'Jul': 'Jul',
        'Agu': 'Aug',
        'Sep': 'Sep',
        'Okt': 'Oct',
        'Nov': 'Nov',
        'Des': 'Dec'
    }

    df = df.copy()

    df['Date'] = df['Date'].astype(str)

    for indo, eng in month_mapping.items():

        df['Date'] = df['Date'].str.replace(
            indo,
            eng,
            regex=False
        )

    df['Date'] = pd.to_datetime(
        df['Date'],
        format='%b-%y'
    )

    df['Date'] = (
        df['Date']
        .dt.to_period('M')
        .dt.to_timestamp()
    )

    df = df.rename(
        columns={
            'Date': 'ds',
            'Sales': 'y'
        }
    )

    df['y'] = pd.to_numeric(
        df['y'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['y']
    )

    return df

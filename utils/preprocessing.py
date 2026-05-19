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

    df['Date'] = df['Date'].astype(str)

    for indo, eng in month_mapping.items():

        df['Date'] = df['Date'].str.replace(
            indo,
            eng
        )

    # convert format Jan-21
    df['Date'] = pd.to_datetime(
        df['Date'],
        format='%b-%y'
    )

    # ubah jadi awal bulan
    df['Date'] = df['Date'].dt.to_period('M')

    df['Date'] = df['Date'].dt.to_timestamp()

    df = df.rename(columns={

        'Date': 'ds',

        'Sales': 'y'

    })

    return df

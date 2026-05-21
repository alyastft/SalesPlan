import pandas as pd
import numpy as np

from scipy.stats import linregress


def classify_items(df):

    classification_results = []

    models = df['Model'].unique()

    for model_name in models:

        item_df = df[
            df['Model'] == model_name
        ].copy()

        item_df = item_df.sort_values('ds')

        sales = item_df['y'].values

        mean_sales = np.mean(sales)

        std_sales = np.std(sales)

        zero_ratio = (sales == 0).mean()

        recent_12 = sales[-12:]

        if mean_sales != 0:
            cv = std_sales / mean_sales
        else:
            cv = 999

        x = np.arange(len(sales))

        slope, _, _, _, _ = linregress(x, sales)

        category = 'Stable'

        if (recent_12 == 0).all():

            category = 'Discontinued'

        elif zero_ratio > 0.3:

            category = 'Intermittent'

        elif slope < -3:

            category = 'Declining'

        elif cv > 1:

            category = 'Volatile'

        else:

            category = 'Stable'

        classification_results.append({

            'Model': model_name,

            'KYB No': item_df['KYB No'].iloc[0],

            'Mean Sales': round(mean_sales, 2),

            'CV': round(cv, 2),

            'Zero Ratio': round(zero_ratio, 2),

            'Trend Slope': round(slope, 2),

            'Category': category

        })

    return pd.DataFrame(classification_results)

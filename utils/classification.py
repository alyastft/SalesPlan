import pandas as pd
import numpy as np
from scipy.stats import linregress


def classify_items(df):

    classification_results = []

    models = df['Model'].unique()

    for model_name in models:

        item_df = df[df['Model'] == model_name].copy()
        item_df = item_df.sort_values('ds')

        sales = item_df['y'].values

        # safety check
        if len(sales) < 3:
            continue

        # NORMALISASI
        if np.max(sales) != np.min(sales):
            sales_norm = (sales - np.min(sales)) / (np.max(sales) - np.min(sales))
        else:
            sales_norm = sales


        # BASIC METRICS (RAW + NORM MIXED SECARA SENGAJA)
        mean_sales = np.mean(sales)
        std_sales = np.std(sales)

        zero_ratio = (sales == 0).mean()

        # aman untuk data kecil
        recent_12 = sales_norm[-12:] if len(sales_norm) >= 12 else sales_norm

        # CV pakai raw (lebih meaningful untuk bisnis)
        cv = std_sales / (mean_sales + 1e-9
                          
        # TREND (NORMALIZED SLOPE)
        x = np.arange(len(sales_norm))

        try:
            slope, _, _, _, _ = linregress(x, sales_norm)
        except:
            slope = 0

        slope = np.clip(slope, -1, 1
                        
        # CLASSIFICATION LOGIC
        category = 'Stable'

        if len(recent_12) > 0 and (recent_12 == 0).all():
            category = 'Discontinued'

        elif zero_ratio > 0.3:
            category = 'Intermittent'

        elif slope < -0.2:
            category = 'Declining'

        elif slope > 0.2:
            category = 'Growing'

        elif cv > 1:
            category = 'Volatile'

        else:
            category = 'Stable'

        # OUTPUT
        classification_results.append({

            'Model': model_name,
            'KYB No': item_df['KYB No'].iloc[0],

            'Mean Sales': round(mean_sales, 2),
            'CV': round(cv, 2),
            'Zero Ratio': round(zero_ratio, 2),

            'Trend Slope': round(slope, 4),

            'Category': category
        })

    return pd.DataFrame(classification_results)

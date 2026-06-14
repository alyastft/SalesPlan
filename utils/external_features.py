import pandas as pd


def load_external_features():

    usd = pd.read_excel(
        "assets/KursUSD.xlsx"
    )

    jpy = pd.read_excel(
        "assets/KursJPY.xlsx"
    )

    motor = pd.read_excel(
        "assets/ProduksiSepedaMotorSeluruh.xlsx"
    )

    usd["Tanggal"] = pd.to_datetime(
        usd["Tanggal"]
    )

    jpy["Tanggal"] = pd.to_datetime(
        jpy["Tanggal"]
    )

    motor["Bulan"] = pd.to_datetime(
        motor["Bulan"]
    )

    usd["ds"] = usd["Tanggal"].dt.to_period(
        "M"
    ).dt.to_timestamp()

    jpy["ds"] = jpy["Tanggal"].dt.to_period(
        "M"
    ).dt.to_timestamp()

    motor["ds"] = motor["Bulan"].dt.to_period(
        "M"
    ).dt.to_timestamp()

    usd = usd[
        ["ds", "Kurs USD/IDR"]
    ]

    jpy = jpy[
        ["ds", "Kurs JPY/IDR"]
    ]

    motor = motor[
        [
            "ds",
            "Domestik",
            "Ekspor"
        ]
    ]

    external_df = (
        usd
        .merge(
            jpy,
            on="ds",
            how="outer"
        )
        .merge(
            motor,
            on="ds",
            how="outer"
        )
    )

    return external_df

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


def mean_absolute_percentage_error(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    return np.mean(np.abs((y - yhat) / y)) * 100


def prepare_data():
    confirmed = pd.read_csv('time_series_19-covid-Confirmed.csv')
    deaths = pd.read_csv('time_series_19-covid-Deaths.csv')
    recovered = pd.read_csv('time_series_19-covid-Recovered.csv')

    for df in [confirmed, deaths, recovered]:
        df.loc[df["Province/State"].isna(), 'Province/State'] = df.loc[df["Province/State"].isna(), 'Country/Region']

    confs = {confirmed.iloc[i, 0] : confirmed.iloc[i, 4:].reset_index().rename(
        columns={"index": "ds", 0: "y"}) for i in range(confirmed.shape[0])}
    death = {deaths.iloc[i, 0] : deaths.iloc[i, 4:].reset_index().rename(
        columns={"index": "ds", 0: "y"}) for i in range(deaths.shape[0])}
    recs = {recovered.iloc[i, 0] : recovered.iloc[i, 4:].reset_index().rename(
        columns={"index": "ds", 0: "y"}) for i in range(recovered.shape[0])}

    confs['Global'] = confirmed.iloc[:, 4:].sum(axis=0).reset_index().rename(columns={"index": "ds", 0: "y"})
    death['Global'] = deaths.iloc[:, 4:].sum(axis=0).reset_index().rename(columns={"index": "ds", 0: "y"})
    recs['Global'] = recovered.iloc[:, 4:].sum(axis=0).reset_index().rename(columns={"index": "ds", 0: "y"})

    return (confs, death, recs)


def predict_n_days(n, df):
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=n)
    forecast = m.predict(future)
    return forecast['yhat'][-1]


def generate_map(df):
    pass


if __name__ == "__main__":
    confirmed = pd.read_csv('time_series_19-covid-Confirmed.csv')
    deaths = pd.read_csv('time_series_19-covid-Deaths.csv')
    recovered = pd.read_csv('time_series_19-covid-Recovered.csv')

    for df in [confirmed, deaths, recovered]:
        df.loc[df["Province/State"].isna(), 'Province/State'] = df.loc[df["Province/State"].isna(), 'Country/Region']

    (conf, death, recs) = prepare_data()
    fs = pd.DataFrame()

    for (location, df) in conf:
        forecast = predict_n_days(7, df)
        forecast[['ds', 'yhat']][-7:]
        forecast = forecast[['ds', 'yhat']][-7:].T
        forecast.columns = forecast.iloc[0]
        forecast = forecast.drop(forecast.index[0])
        forecast.insert(0, 'Location', location)
        if location != 'Global':
            forecast.insert(1, 'Lat', confirmed.loc[confirmed['Province/State'] == location, 'Lat'].item())
            forecast.insert(2, 'Long', confirmed.loc[confirmed['Province/State'] == location, 'Long'].item())
        else:
            forecast.insert(1, 'Lat', None)
            forecast.insert(2, 'Long', None)
        fs = fs.append(forecast)

    generate_map(fs)

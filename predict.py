import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import plotly.express as px
import os
from datetime import datetime


def prepare_data():
    confirmed = pd.read_csv('time_series_19-covid-Confirmed.csv')
    deaths = pd.read_csv('time_series_19-covid-Deaths.csv')
    recovered = pd.read_csv('time_series_19-covid-Recovered.csv')

    for df in [confirmed, deaths, recovered]:
        df.loc[df["Province/State"].isna(), 'Province/State'] = df.loc[df["Province/State"].isna(), 'Country/Region']

    confs = {}
    death = {}
    recs = {}
    for i in range(confirmed.shape[0]):
        a = confirmed.iloc[i, 4:].reset_index()
        a.columns = ['ds', 'y']
        confs[confirmed.iloc[i, 0]] = a
    for i in range(deaths.shape[0]):
        a = deaths.iloc[i, 4:].reset_index()
        a.columns = ['ds', 'y']
        death[deaths.iloc[i, 0]] = a
    for i in range(recovered.shape[0]):
        a = recovered.iloc[i, 4:].reset_index()
        a.columns = ['ds', 'y']
        recs[recovered.iloc[i, 0]] = a

    confs['Global'] = confirmed.iloc[:, 4:].sum(axis=0).reset_index()
    confs['Global'].columns = ['ds', 'y']
    death['Global'] = deaths.iloc[:, 4:].sum(axis=0).reset_index()
    death['Global'].columns = ['ds', 'y']
    recs['Global'] = recovered.iloc[:, 4:].sum(axis=0).reset_index()
    recs['Global'].columns = ['ds', 'y']

    return (confs, death, recs)


def predict_n_days(n, df):
    with suppress_stdout_stderr():
        m = Prophet(daily_seasonality=True)
        m.fit(df)
    future = m.make_future_dataframe(periods=n)
    forecast = m.predict(future)
    return forecast


def generate_map(df):
    newDF = df.copy()
    dates = newDF.columns[4:]
    df2 = pd.DataFrame(columns=['Place', 'Lat', 'Long', 'Date', 'Size', 'Text', 'Color'])
    for place in newDF["Location"].unique():
        for date in dates:
            Lat = df.loc[df["Location"] == place, "Lat"]
            Long = df.loc[df["Location"] == place, "Long"]
            number = df.loc[df["Location"] == place, date].item()
            text = "Number of cases: " + \
                str(df.loc[df["Location"] == place,
                           date].item()) + " in " + str(place)
            size = (np.log(
                int(df.loc[df["Location"] == place, date].item()))/np.log(1e15))*500
            if (np.isinf(size)):
                size = 0
            if (number < 3):
                color = "< 3"
            elif (3 <= number < 11):
                color = "< 11"
            elif (11 <= number < 21):
                color = "< 21"
            elif (21 <= number < 51):
                color = "< 51"
            elif (number > 50):
                color = "50+"
            temp = pd.DataFrame({"Place": [place],
                                 "Lat": int(Lat),
                                 "Long": int(Long),
                                 "Date": date,
                                 "Size": int(size),
                                 "Text": str(text),
                                 "Color": color})
            df2 = df2.append(temp)
    fig = px.scatter_geo(df2, lat="Lat", lon="Long", color="Color",
                         hover_name="Place", size=(list(df2["Size"])),
                         animation_frame="Date",
                         text="Text",
                         locationmode='country names',
                         projection="natural earth")
    return fig


# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


if __name__ == "__main__":
    confirmed = pd.read_csv('time_series_19-covid-Confirmed.csv')
    deaths = pd.read_csv('time_series_19-covid-Deaths.csv')
    recovered = pd.read_csv('time_series_19-covid-Recovered.csv')

    for df in [confirmed, deaths, recovered]:
        df.loc[df["Province/State"].isna(), 'Province/State'] = df.loc[df["Province/State"].isna(), 'Country/Region']

    (conf, death, recs) = prepare_data()
    fs = pd.DataFrame()

    for (location, df) in conf.items():
        forecast = predict_n_days(7, df)
        forecast[['ds', 'yhat']][-7:]
        forecast = forecast[['ds', 'yhat']][-7:].T
        forecast.columns = map(lambda t: t.strftime('%-m/%-d/%y'), forecast.iloc[0])
        forecast = forecast.drop(forecast.index[0])
        forecast.insert(0, 'Location', location)
        if location != 'Global':
            forecast.insert(1, 'Lat', confirmed.loc[confirmed['Province/State'] == location, 'Lat'].item())
            forecast.insert(2, 'Long', confirmed.loc[confirmed['Province/State'] == location, 'Long'].item())
        else:
            forecast.insert(1, 'Lat', None)
            forecast.insert(2, 'Long', None)
        fs = fs.append(forecast)
        print(len(fs))

    fig = generate_map(fs.iloc[:-1])
    fig.show()

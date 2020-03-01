import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
import plotly.express as px
import os
from datetime import datetime

c_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
d_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
r_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'

def prepare_data(confirmed, deaths, recovered):
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


def generate_map(df, curr=False):
    newDF = df.copy()
    if curr:
        dates = newDf.columns[4:]
    else:
        dates = newDF.columns[3:]
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
                         hover_data=["Date"],
                         locationmode='country names',
                         projection="natural earth")
    return fig


def refresh(ref_type):
    confirmed = pd.read_csv(c_url)
    deaths = pd.read_csv(d_url)
    recovered = pd.read_csv(r_url)

    for df in [confirmed, deaths, recovered]:
        df.loc[df["Province/State"].isna(), 'Province/State'] = df.loc[df["Province/State"].isna(), 'Country/Region']

    (conf, death, recs) = prepare_data(confirmed, deaths, recovered)
    fs = pd.DataFrame()

    if ref_type == 'confirmed':
        for (location, df) in conf.items():
            forecast = predict_n_days(28, df)
            forecast[['ds', 'yhat']][-28:]
            forecast = forecast[['ds', 'yhat']][-28:].T
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
            print('confirmed, ' + str(len(forecast)))
        fig = generate_map(fs.iloc[:-1])
        fig.write_html('templates/fig_c.html')
    elif ref_type == 'deaths':
        for (location, df) in death.items():
            forecast = predict_n_days(28, df)
            forecast[['ds', 'yhat']][-28:]
            forecast = forecast[['ds', 'yhat']][-28:].T
            forecast.columns = map(lambda t: t.strftime(
                '%-m/%-d/%y'), forecast.iloc[0])
            forecast = forecast.drop(forecast.index[0])
            forecast.insert(0, 'Location', location)
            if location != 'Global':
                forecast.insert(
                    1, 'Lat', confirmed.loc[confirmed['Province/State'] == location, 'Lat'].item())
                forecast.insert(
                    2, 'Long', confirmed.loc[confirmed['Province/State'] == location, 'Long'].item())
            else:
                forecast.insert(1, 'Lat', None)
                forecast.insert(2, 'Long', None)
            fs = fs.append(forecast)
            print('deaths, ' + str(len(forecast)))
        fig = generate_map(fs.iloc[:-1])
        fig.write_html('templates/fig_d.html')
    elif ref_type == 'recovered':
        for (location, df) in recs.items():
            forecast = predict_n_days(28, df)
            forecast[['ds', 'yhat']][-28:]
            forecast = forecast[['ds', 'yhat']][-28:].T
            forecast.columns = map(lambda t: t.strftime(
                '%-m/%-d/%y'), forecast.iloc[0])
            forecast = forecast.drop(forecast.index[0])
            forecast.insert(0, 'Location', location)
            if location != 'Global':
                forecast.insert(
                    1, 'Lat', confirmed.loc[confirmed['Province/State'] == location, 'Lat'].item())
                forecast.insert(
                    2, 'Long', confirmed.loc[confirmed['Province/State'] == location, 'Long'].item())
            else:
                forecast.insert(1, 'Lat', None)
                forecast.insert(2, 'Long', None)
            fs = fs.append(forecast)
            print('recovered, ' + str(len(forecast)))
        fig = generate_map(fs.iloc[:-1])
        fig.write_html('templates/fig_r.html')
    elif ref_type == 'curr_confirmed':
        fig = generate_map(confirmed, curr=True)
        fig.write_html('templates/curr_c.html')
    elif ref_type == 'curr_deaths':
        fig = generate_map(deaths, curr=True)
        fig.write_html('templates/curr_d.html')
    elif ref_type == 'curr_recovered':
        fig = generate_map(recovered, curr=True)
        fig.write_html('templates/curr_r.html')


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

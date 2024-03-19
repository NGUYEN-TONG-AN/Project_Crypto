import pandas as pd
import sqlite3
import pickle
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
import numpy as np
from datetime import datetime, timedelta
#Biliothèque pour Fast api et Dash:
import dash
from fastapi import FastAPI, Depends, HTTPException, Cookie, Form, Request
from fastapi.applications import FastAPI as FastAPIApp
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.security import OAuth2PasswordBearer
from user_management.user_manager import User, create_user, get_user_by_username, delete_user
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import Dash 
#from dash_bootstrap_components import State
from fastapi import Cookie

# Initialiser FastAPI
app_fastapi = FastAPI()

# Initialiser Dash 
app_dash = Dash(__name__, suppress_callback_exceptions=True)

# Initialiser Dash avec enrichissement FastAPI
#app_dash = Dash(__name__, server=app_fastapi, url_base_pathname='/dash/')

#dash_app = app_dash.server

# Initialiser le moteur de modèles Jinja2
templates = Jinja2Templates(directory="templates")

# Chemin vers les fichiers .sav
arima_btc_eur_path = "/home/ubuntu/Project_Crypto/models/ARIMA_BTC_EUR.sav"
arima_btc_usd_path = "/home/ubuntu/Project_Crypto/models/ARIMA_BTC_USD.sav"
arima_eth_usd_path = "/home/ubuntu/Project_Crypto/models/ARIMA_ETH_USD.sav"
lstm_btc_eur_path = "/home/ubuntu/Project_Crypto/models/LSTM_BTC_EUR.sav"
lstm_btc_usd_path = "/home/ubuntu/Project_Crypto/models/LSTM_BTC_USD.sav"
lstm_eth_usd_path = "/home/ubuntu/Project_Crypto/models/LSTM_ETH_USD.sav"

# Charger les modèles ARIMA
arima_btc_eur_model = pickle.load(open(arima_btc_eur_path, 'rb'))
arima_btc_usd_model = pickle.load(open(arima_btc_usd_path, 'rb'))
arima_eth_usd_model = pickle.load(open(arima_eth_usd_path, 'rb'))

# Charger les modèles LSTM
lstm_btc_eur_model = pickle.load(open(lstm_btc_eur_path, 'rb'))
lstm_btc_usd_model = pickle.load(open(lstm_btc_usd_path, 'rb'))
lstm_eth_usd_model = pickle.load(open(lstm_eth_usd_path, 'rb'))

# Se connecter à la base de données
database_path = "/home/ubuntu/Project_Crypto/Database/base_de_donnees.db"
conn = sqlite3.connect(database_path)

# Charger les données de streaming depuis la base de données
query = "SELECT * FROM Donnees_Streaming"
streaming_data = pd.read_sql(query, conn)

# Fermer la connexion à la base de données
conn.close()

# Diviser les données en fonction de la colonne 'ID PAIRE'
btc_usd_data = streaming_data[streaming_data['ID PAIRE'] == 1].copy()
btc_eur_data = streaming_data[streaming_data['ID PAIRE'] == 2].copy()
eth_usd_data = streaming_data[streaming_data['ID PAIRE'] == 3].copy()

# Colonnes nécessaires pour la prédiction
features_columns = ['Close', 'Volume', 'High', 'Low', 'Open']

# Fonction pour faire la prédiction avec le temps choisi par utilisateur:----------------------------------
def extend_predictions_arima_lstm(data, arima_model, lstm_model, t):
    # Utiliser ARIMA pour étendre les prédictions/////////////////////////////////////////////////////////
    last_date = data.index[-1]
 # Calculer la date suivante
    next_date = last_date + pd.DateOffset(days=t)
 
    # Faire une prédiction pour le jour suivant
    forecast_next_day = arima_model.forecast(steps=t)

    # Créer une nouvelle ligne pour la date suivante avec la prédiction
    new_row = pd.Series(index=data.columns, name=next_date)
    new_row['Forecast'] = forecast_next_day[0]
   
    # Ajouter la nouvelle ligne au DataFrame
 
    data = pd.concat([data, new_row.to_frame().transpose()], ignore_index=False)
    data['Forecast'].iloc[-2] = data['Close'].iloc[-2]
    data.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)

    # Utiliser LSTM pour étendre les prédictions//////////////////////////////////////////////////
    df = data.copy() #modifier  data  par data.copy() pour afficher les donnees reel de j-7
    y = df['Close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = t  # length of output sequences (forecast period)
    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = lstm_model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-7] + pd.Timedelta(days=7), periods=n_forecast) #modifier iloc[-1],days=1 par >
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    results = pd.concat([df_past, df_future], ignore_index=True)
    results.set_index('Date', inplace=True)

    return results, data 
# Fin fonction----------------------------------------------------------------------------------------------
# Charger la base d'utilisateurs
user_manager = User()


# OAuth2 pour la gestion de l'authentification via FastAPI
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

#Creation page html : html*******************************************************************************************
# Page de connexion

#Fin les 2 pages html  :html **********************************************************************************************

# Route pour la page de connexion
@app_fastapi.get("/")
def login(request: Request, user_id: str = Cookie(None)):
    return HTMLResponse(content="ok", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_fastapi, host="127.0.0.1", port=8000, debug = True) 
#Fin page prédiction ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))


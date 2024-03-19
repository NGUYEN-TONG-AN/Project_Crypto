import pandas as pd
import sqlite3
import pickle
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Sequence
#Biliothèque pour Fast api et Dash:
import json
import dash
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, HTTPException, Cookie, Form, Request, Query
from fastapi.applications import FastAPI as FastAPIApp
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.security import OAuth2PasswordBearer
#from werkzeug.middleware.dispatcher import DispatcherMiddleware
#from user_management.user_manager import User, Base, create_user, get_user_by_username, delete_user, get_db
#from dash import Dash, dcc,  html, no_update

#from dash.dependencies import Input, Output, State
#from dash_bootstrap_components import State
from fastapi import Cookie

app_fastapi = FastAPI()

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
query_historique = "SELECT * FROM Donnée_historique"
donnees_historique = pd.read_sql(query_historique, conn)
# Fermer la connexion à la base de données
conn.close()

# Diviser les données en fonction de la colonne 'ID PAIRE'
btc_usd_data = streaming_data[streaming_data['ID PAIRE'] == 1].copy()
btc_eur_data = streaming_data[streaming_data['ID PAIRE'] == 2].copy()
eth_usd_data = streaming_data[streaming_data['ID PAIRE'] == 3].copy()
# Diviser les données hostorique en fonction de la colonne 'ID PAIRE'
btc_usd_data_historique = donnees_historique[donnees_historique['ID PAIRE'] == 1].copy()
btc_eur_data_historique = donnees_historique[donnees_historique['ID PAIRE'] == 2].copy()
eth_usd_data_historique = donnees_historique[donnees_historique['ID PAIRE'] == 3].copy()
#Faire les copies les 60lignes des donnees historique si les donnees streaming < 60 lignes pour LSTM:
# Copier les 60 dernières lignes de btc_usd_data_historique à btc_usd_data si nécessaire
btc_usd_data_historique_tail = btc_usd_data_historique.tail(60)
if len(btc_usd_data) < 60:
    btc_usd_data = pd.concat([btc_usd_data_historique_tail[::-1], btc_usd_data])

# Répéter le processus pour les autres paires
btc_eur_data_historique_tail = btc_eur_data_historique.tail(60)
if len(btc_eur_data) < 60:
    btc_eur_data = pd.concat([btc_eur_data_historique_tail[::-1], btc_eur_data])

eth_usd_data_historique_tail = eth_usd_data_historique.tail(60)
if len(eth_usd_data) < 60:
    eth_usd_data = pd.concat([eth_usd_data_historique_tail[::-1], eth_usd_data])
 
# Colonnes nécessaires pour la prédiction
features_columns = ['Close', 'Volume', 'High', 'Low', 'Open']

# Fonction pour faire la prédiction avec le temps choisi par utilisateur:----------------------------------
def extend_predictions_arima_lstm(data, arima_model, lstm_model, t):
    # Utiliser ARIMA pour étendre les prédictions/////////////////////////////////////////////////////////
    #last_date = data.index[-1]
    data=data.drop('Date_NP',axis = 1)     
    last_date = data.iloc[-1]['Date']
    print("Last date :",last_date)
    current_date = datetime.now().date()
    #selected_timedelta = current_date + timedelta(days=t)
    #selected_timedelta_1 = datetime.combine(selected_timedelta, datetime.min.time())
    # Calculer la date suivante
    next_date = last_date + pd.DateOffset(days=t)
    print("Next dates :",next_date)
    # Faire une prédiction pour le jour suivant
    forecast_next_day = arima_model.forecast(steps=t)
    #forecast_next_day = forecast_next_day.reset_index(drop=True)
    #prendre les valeurs impaire dans la liste forecast_next_day
    forecast_values = forecast_next_day.tolist()

    # Créer une nouvelle ligne pour la date suivante avec la prédiction
    #new_row = pd.Series(index=data.columns, name=next_date)
    print("forecast_values :",forecast_values)
    print("Forecast next day:",forecast_next_day)
    #new_row['Forecast'] = forecast_next_day[0:t]
    
    #print(new_row)
    # Ajouter la nouvelle ligne au DataFrame
    #print("New Row da ta frame apres  :",new_row.head())
    #new_row = pd.DataFrame(new_row)
    #print("New Row da ta frame apres  :",new_row.head())
    #data = pd.concat([data, new_row.transpose()], ignore_index=False)
    #data['Forecast'].iloc[-2] = data['Close'].iloc[-2]
    #print("12 ieme ligne",data.head(-1))
    #print("data['Close'].iloc[-2] :",data['Close'].iloc[-2])
    data['Forecast'] = np.nan
    if len(forecast_values) > 0:
    # Ajouter les valeurs dans la colonne 'Forecast' des dernières lignes du DataFrame data
        temp_df = pd.DataFrame(index=pd.date_range(start=current_date + pd.Timedelta(days=1), periods=len(forecast_values)), columns=['Forecast'])
        temp_df['Forecast'] = forecast_values
        temp_df['Date'] = temp_df.index
        temp_df=temp_df.reset_index(drop = True)
        data = pd.concat([data, temp_df], ignore_index=True)
    else:
    # Gérer le cas où le nombre de valeurs est 0
     	print("Aucune valeur de prévision à ajouter.")
    print("data apres rajoute forecast :",data['Forecast'])
    print("data df :",temp_df)
    #data.loc[data.index[-2], 'Forecast'] = data['Close'].iloc[-2]
    #print("data['Close'].iloc[-2] :",data['Close'].iloc[-2])
    data.rename(columns={'Close': 'Actual'}, inplace=True)
    results_arima =  data
    print("Data Arima :",results_arima)
    # Utiliser LSTM pour étendre les prédictions//////////////////////////////////////////////////
    df = data.copy() #modifier  data  par data.copy() pour afficher les donnees reel de j-7
    y = df['Actual'].fillna(method='ffill')
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
    for i in range(t - 1):
    # Utilisation de la prédiction précédente comme entrée pour la prédiction suivante
        #next_input = np.append(X_[-1, -1, 0], Y_[-1])  # Concaténer la prédiction précédente à la dernière séquence
        next_input = np.append(X_[-1, -n_lookback+1:, 0], Y_[-1])
        next_input = next_input.reshape(1, n_lookback, 1)
        next_prediction = lstm_model.predict(next_input).reshape(-1, 1)
        next_prediction = scaler.inverse_transform(next_prediction)
    
       # Ajout de la prédiction à Y_
        Y_ = np.concatenate([Y_, next_prediction])
        X_ = np.concatenate([X_, next_input])
    
    # organize the results in a data frame
    df_past = df[['Actual']].reset_index()
    
    #df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df['Date'])
    df_past['Forecast'] = np.nan
    #df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    #df_past['Forecast'].[-1:-t] = df['Actual'][-1:-t]
    print("DF PAST :",df_past['Date'].head())
    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_past.iloc[-t:, df_past.columns.get_loc('Forecast')] = Y_.flatten()
    #df_future['Date'] = df_past['Date'].iloc[- (t+7):]  #modifier iloc[-1],days=1 par iloc[-7],days=7 pour les 7derier jour de valeur  reel et prédiction
    print("DF PAST1 :",df_past)
    #print("Y_flatten :",Y_.flatten())
    #df_future['Forecast'] = Y_.flatten()
    #df_future.iloc[-t:, df_future.columns.get_loc('Forecast')] = Y_.flatten()
    #df_future['Actual'] = np.nan
    #print("DF FUTURE2 :",df_future)
    #results_lstm = pd.concat([df_past, df_future], ignore_index=True)
    #results_lstm.set_index('Date', inplace=True)
    results_lstm = df_past
    print("RESULT LSTM Actual :",results_lstm)
   
    
    return results_lstm, results_arima 
#Fin de fonction------------------------------------------------------------------------------------------
# Fonction update les figures pour les 2 modèle prédiction avec le temps:

def update_plots(crypto_value, selected_days):
    # Convertir la date sélectionnée en objet datetime
    #selected_date = datetime.strptime(selected_date, '%Y-%m-%d')
    #selected_timedelta = timedelta(days=selected_days)
    current_date = datetime.now().date()
    selected_timedelta = current_date + timedelta(days=selected_days)
    selected_timedelta = datetime.combine(selected_timedelta, datetime.min.time())
    print('selected date :',selected_timedelta)
    # Sélectionner les données appropriées en fonction de la crypto choisie
    if crypto_value == 'btc_usd':
        crypto_data = btc_usd_data
        arima_model = arima_btc_usd_model
        lstm_model = lstm_btc_usd_model
    elif crypto_value == 'btc_eur':
        crypto_data = btc_eur_data
        arima_model = arima_btc_eur_model
        lstm_model = lstm_btc_eur_model
    elif crypto_value == 'eth_usd':
        crypto_data = eth_usd_data
        arima_model = arima_eth_usd_model
        lstm_model = lstm_eth_usd_model
    else:
        # Valeur crypto non reconnue, renvoyer des figures
        return {}, {}
    #Convertir la colonne date en type date time

    #selected_date = datetime.strptime(str(crypto_data.iloc[-1]['Date']), '%Y-%m-%d')
    selected_date = crypto_data.iloc[-1]['Date']
    print('date selectionne :' ,selected_date)
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date'])
    #Filtrer les données jusqu'à la date sélectionnée
    filtered_data = crypto_data[crypto_data['Date'] <= selected_date]
    print('data filtre :',filtered_data)
    print('Selected time delta :',selected_timedelta)
    
    # Calculer la différence en jours entre la dernière date dans les données réelle>
    #t = (selected_timedelta + selected_date - datetime.now()).days
    #print('valeur du t :',t)
    # Faire la prédiction avec la fonction extend_predictions_arima_lstm
    #results_lstm, results_arima  = extend_predictions_arima_lstm(filtered_data, arima_model, lstm_model,selected_days )///TEST
    results_lstm, results_arima  = extend_predictions_arima_lstm(crypto_data, arima_model, lstm_model,selected_days )
    #Nettoyer les donnees : Enlever les Nan dans le data frame:
    results_lstm['Forecast'].fillna('',inplace = True)
    results_lstm['Actual'].fillna('',inplace = True)
    results_arima['Forecast'].fillna('',inplace = True)
    results_arima['Actual'].fillna('',inplace = True)
    print("data arima :",results_arima)
    print("data arima :",results_lstm)
    # Créer les figures pour les deux graphiques
    arima_fig = {
        'data': [
            {'x': results_arima['Date'], 'y': results_arima['Actual'], 'name': 'Actual', 'mode': 'lines'},
            {'x': results_arima['Date'], 'y': results_arima['Forecast'], 'name': 'ARIMA Forecast', 'mode': 'lines'}
        ],
        'layout': {'title':  f'ARIMA Prediction pour {crypto_value} pour Jours {selected_days}'}
    }
    lstm_fig = {
        'data': [
            {'x': results_lstm['Date'], 'y': results_lstm['Actual'], 'name': 'Actual', 'mode': 'lines'},
            {'x': results_lstm['Date'], 'y': results_lstm['Forecast'], 'name': 'LSTM Forecast', 'mode': 'lines'}
        ],
        'layout': {'title': f'LSTM Prediction pour {crypto_value} pour Jours {selected_days}'}
    }
    arima_date = results_arima['Date'].tolist()
    arima_actual = results_arima['Actual'].tolist()
    arima_forecast = results_arima['Forecast'].tolist()
    # Convertir les objets Pandas Series en listes
    arima_data_list = [
    {0: arima_date, 1: [float(value) if value != '' else 0 for value in arima_actual], 'name': 'Actual', 'mode': 'lines+markers'},
    {0: arima_date, 1: [float(value) if value != '' else 0 for value in arima_forecast], 'name': 'ARIMA Forecast', 'mode': 'lines+markers'}
                      ]
    # Créer la figure pour le graphique ARIMA
    arima_fig = {
    0: arima_data_list,
    1: {'title': f'ARIMA Prediction pour {crypto_value} pour Jours {selected_days}'}
                }
    # Sérialiser la figure en JSON
    arima_json = json.dumps(arima_fig, default=str)
    lstm_date = results_lstm['Date'].tolist()
    lstm_actual = results_lstm['Actual'].tolist()
    lstm_forecast = results_lstm['Forecast'].tolist()
    # Convertir les objets Pandas Series en listes
    lstm_data_list = [
    {0: lstm_date, 1: [float(value) if value != '' else 0 for value in lstm_actual], 'name': 'Actual', 'mode': 'lines+markers'},
    {0: lstm_date, 1: [float(value) if value != '' else 0 for value in lstm_forecast], 'name': 'LSTM Forecast', 'mode': 'lines+markers'}
                      ]
    # Créer la figure pour le graphique ARIMA
    lstm_fig = {
    0: lstm_data_list,
    1: {'title': f'LSTM Prediction pour {crypto_value} pour Jours {selected_days}'}
                }
    lstm_json = json.dumps(lstm_fig, default=str)
    #return arima_fig, lstm_fig
    return arima_json, lstm_json
#Fin de fonction -------------------------------------------------------------------------------------------------------------
#Route API pour la prédiction:

@app_fastapi.get("/prediction", response_class=HTMLResponse)
async def prediction_endpoint(symbol: str = Query(..., title="Symbol of Cryptocurrency"),
                              t: int = Query(..., title="Time for Prediction (J+1, J+2, etc.)")):
    # Appeler la fonction d'extension des prédictions avec les modèles ARIMA et LSTM
    arima_fig, lstm_fig = update_plots(symbol, t)
    # Extraire les valeurs de symbol et t pour les inclure dans la chaîne de format JavaScript
    symbol_str = str(symbol)
    t_str = str(t)
    #Return type Json 
    #return {"symbol": symbol, "time": t, "arima_plot": arima_fig, "lstm_plot": lstm_fig}
    #Return type Html pour les plots
    # Extraire les données du graphique ARIMA
    arima_data = json.loads(arima_fig)
    arima_actual_data = arima_data['0'][0]
    arima_forecast_data = arima_data['0'][1]
    #arima_data_list = arima_data[0]
    #arima_data = [list(arima_data[col]) for col in arima_data.columns] if isinstance(arima_data, pd.Series) else arima_data
    #arima_data_json = json.dumps(arima_data)
    # Convertir les colonnes de dates et numériques en listes Python
    # Extraire les données du graphique ARIMAreturn arima_json, lstm_json
    
    lstm_data = json.loads(lstm_fig)
    
    lstm_actual_data = lstm_data['0'][0]
    lstm_forecast_data = lstm_data['0'][1]

# Extraire les données du graphique LSTM
    #lstm_data_list = lstm_data[0]
    #arima_layout = arima_data[1]
    
    
    
    #lstm_data = [list(lstm_data[col]) for col in lstm_data.columns] if isinstance(lstm_data, pd.Series) else lstm_data
    #lstm_data_json = json.dumps(lstm_data)
    
    
    
    #lstm_layout = lstm_data[1]

    html_content=f"""
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div>
                    <h1>ARIMA Prediction for {symbol} at Time {t}</h1>
                    <p>Test ARIMA content</p>
                    <div id="arima-plot" style="min-height: 300px;"></div>
                </div>
                <div>
                    <h1>LSTM Prediction for {symbol} at Time {t}</h1>
                    <p>Test LSTM content</p>
                    <div id="lstm-plot" style="min-height: 300px;"></div>
                </div>
                <script>
                    // Afficher le graphique ARIMA
                    var arimaActualData  = {arima_actual_data};
                    var arimaForecastData = {arima_forecast_data};
                     Plotly.newPlot('arima-plot', [
                        {{x: arimaActualData['0'], y: arimaActualData['1'], name: 'Actual', mode: 'lines+markers'}},
                        {{x: arimaForecastData['0'], y: arimaForecastData['1'], name: 'ARIMA Forecast', mode: 'lines+markers'}}
                    ]);



                    // Afficher le graphique LSTM
                    var lstmActualData = {lstm_actual_data};
                    var lstmForecastData = {lstm_forecast_data};
                    Plotly.newPlot('lstm-plot', [
                        {{x: lstmActualData['0'], y: lstmActualData['1'], name: 'Actual', mode: 'lines+markers'}},
                        {{x: lstmForecastData['0'], y: lstmForecastData['1'], name: 'LSTM Forecast', mode: 'lines+markers'}}
                    ]);
                </script>
            </body>
        </html>
    """
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
   # app_fastapi.add_middleware(WSGIMiddleware, app_dash.server)
    #app_dash.run_server(host="0.0.0.0", port=8000, mode="external")    
    import uvicorn
    uvicorn.run(app_fastapi, host="127.0.0.1", port=8000)
    #app_dash.run_server(debug=True)  

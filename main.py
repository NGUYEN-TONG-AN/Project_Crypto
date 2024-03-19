import pandas as pd
import sqlite3
import pickle
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Sequence
#Biliothèque pour Fast api et Dash:
import dash
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, HTTPException, Cookie, Form, Request
from fastapi.applications import FastAPI as FastAPIApp
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.security import OAuth2PasswordBearer
from user_management.user_manager import User, Base, create_user, get_user_by_username, delete_user, get_db
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import Dash 
#from dash_bootstrap_components import State
from fastapi import Cookie
#DATABASE_URL = "sqlite:///./user_management/base_utilisateur.db" #  la base de données SQLite sera stockée dans le fic>

#engine = create_engine(DATABASE_URL) # Le moteur est responsable de la communication avec la base de d>

#Base = declarative_base() #La classe de base déclarative fournit des fonctionnalités de base pour décl>
#SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Initialiser FastAPI
app_fastapi = FastAPI()

# Initialiser Dash 
app_dash = Dash(__name__, suppress_callback_exceptions=True)

#Récupérer les utilisateurs de la base de données et les stocker dans une liste ou un dictionnaire
def get_users():
    with get_db() as db:
        users = db.query(User).all()
        print("Users from the database:", users)
        return db.query(User).all()

def get_all_users(db):
    try:
        all_users = db.query(User).all()
        print("Users from the database:", all_users)
        return all_users
    except Exception as e:
        print("Error retrieving users:", e)
        return []

users_data = {user.username: user.password for user in get_users()}
print("users")
print(users_data)
# Fonction pour vérifier les identifiants de l'utilisateur
def verify_credentials(db: Session, provided_username: str, provided_password: str) -> bool:
    # Ajoutez cette ligne pour imprimer la requête SQL
    print(db.query(User).filter(User.username == provided_username).statement)
    #query = db.query(User).filter(User.username == provided_username)
    #user = query.first()
    user = db.query(User).filter(User.username.ilike(provided_username)).first()
     # Log de la requête SQL
   # print("SQL Query:", query)
    print("User from database:", user.username if user else "No user found")
    print("Provided username:", provided_username)
    print("Provided password:", provided_password)
    print("User password:", user.password if user else "No user found")

    if user and user.password == provided_password:
        return True

    return False
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
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-7] + pd.Timedelta(days=7), periods=n_forecast) #modifier iloc[-1],days=1 par iloc[-7],days=7 pour les 7derier jour de valeur  reel et prédiction
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    results = pd.concat([df_past, df_future], ignore_index=True)
    results.set_index('Date', inplace=True)

    return results, data 
# Fin fonction----------------------------------------------------------------------------------------------
# Charger la base d'utilisateurs
user_manager = User()

# OAuth2 pour la gestion de l'authentification via FastAPI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

#Creation page html : html*******************************************************************************************
# Page de connexion
login_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Login Page</title>
</head>
<body>
    <h1>Login</h1>
    <form action="/login" method="post">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required><br>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
     <h4>Veuillez rentrer le bon utilisateur et mdp</h4>
</body>
</html>
"""

# Page d'accueil de Dash
dash_home_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Dash Home Page</title>
</head>
<body>
    <h1>Dash Home</h1>
    <div id="arima-plot"></div>
    <div id="lstm-plot"></div>
</body>
</html>
"""
#Fin les 2 pages html  :html **********************************************************************************************

# Route pour la page de connexion
@app_fastapi.get("/")
async def login(request: Request, user_id: str = Cookie(None)):
    return HTMLResponse(content=login_page, status_code=200)

# Route pour le formulaire de connexion
@app_fastapi.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Vérifier l'identifiant de l'utilisateur
    print("Username:", username)
    print("Password:", password)
    print("DB session in login route:",db)
    #print("DB session:", db)
    try:
        all_users = db.query(User).all()
        print("Users from the database:", all_users)
    except Exception as e:
        print("Error retrieving users:", e)
        all_users = []
    if verify_credentials(db, username, password):
        # Utilisateur authentifié, rediriger vers la page Dash
        response = RedirectResponse(url="/prediction")
        response.set_cookie("user_id", user_manager.get_user_id(username))
        return response
    else:
        # Informations d'identification incorrectes, rediriger vers la page de connexion
        print('not ok')
        all_users = get_all_users(db)
        print("Users from the database:", all_users)

        return HTMLResponse(content=login_page, status_code=401)

# Définir la mise en page de Dash pour la page de prédiction))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
app_dash.layout = html.Div([
    html.H1("Page de Prédiction"),
    dcc.Dropdown(
        id='crypto-dropdown',
        options=[
            {'label': 'BTC-USD', 'value': 'btc_usd'},
            {'label': 'BTC-EUR', 'value': 'btc_eur'},
            {'label': 'ETH-USD', 'value': 'eth_usd'}
        ],
        value='btc_usd'
    ),
    dcc.DatePickerSingle(
        id='prediction-date-picker',
        date=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    ),
    dcc.Graph(id='arima-plot'),
    dcc.Graph(id='lstm-plot'),
    html.Div(dash_home_page, id='dash-home-content')
])
# Route pour la page de prédiction
@app_fastapi.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request, user_id: str = Cookie(None)):
    # Vérifier si l'utilisateur est connecté
    if not user_manager.is_user_authenticated(user_id):
        # Utilisateur non authentifié, rediriger vers la page de connexion
        return RedirectResponse(url="/", status_code=302)

    # Utilisateur authentifié, afficher la page de prédiction
    return app_dash.index()

#Fin page prédiction ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

#DASH pour affichier la dynamic des plots: dashdashdashdashdashdashdashdashdashdashdashdashdashdashdashdashdashdashdash
# Route pour la page de prédiction avec Dash
@app_dash.callback(
    Output('arima-plot', 'figure'),
    Output('lstm-plot', 'figure'),
    Input('crypto-dropdown', 'value'),
    Input('prediction-date-picker', 'date')
)

def update_plots(crypto_value, selected_date):
    # Convertir la date sélectionnée en objet datetime
    selected_date = datetime.strptime(selected_date, '%Y-%m-%d')

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
        # Valeur crypto non reconnue, renvoyer des figures vides
        return {}, {}

    # Filtrer les données jusqu'à la date sélectionnée
    filtered_data = crypto_data[crypto_data.index <= selected_date]

    # Calculer la différence en jours entre la dernière date dans les données réelles et la date de prédiction
    t = (selected_date - crypto_data.index[-1]).days

    # Faire la prédiction avec la fonction extend_predictions_arima_lstm
    predictions, extended_data = extend_predictions_arima_lstm(filtered_data, arima_model, lstm_model, t)

    # Créer les figures pour les deux graphiques
    arima_fig = {
        'data': [
            {'x': extended_data.index, 'y': extended_data['Close'], 'name': 'Actual', 'mode': 'lines'},
            {'x': predictions.index, 'y': predictions['Forecast'], 'name': 'ARIMA Forecast', 'mode': 'lines'}
        ],
        'layout': {'title': 'ARIMA Prediction'}
    }

    lstm_fig = {
        'data': [
            {'x': extended_data.index, 'y': extended_data['Close'], 'name': 'Actual', 'mode': 'lines'},
            {'x': predictions.index, 'y': predictions['Forecast'], 'name': 'LSTM Forecast', 'mode': 'lines'}
        ],
        'layout': {'title': 'LSTM Prediction'}
    }

    return arima_fig, lstm_fig
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)

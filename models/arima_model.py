#Importation des librairies
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import pickle

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

fichier_csv ='/home/ubuntu/Project_Crypto/data_processing/Base_Donnees_final.csv'
df = pd.read_csv(fichier_csv)
# Groupez les données par la colonne "ID PAIRE"
groupes = df.groupby('ID PAIRE')

# Créez trois DataFrames distincts
crypto_BTC_USD_df = groupes.get_group(1)
crypto_BTC_EUR_df = groupes.get_group(2)
crypto_ETH_USD_df = groupes.get_group(3)

# Fonction pour sauvegarder le modèle LSTM
def save_arima_model(model,filename):
   try:
        pickle.dump(model, open(filename, 'wb'))
        print(f"Model saved successfully as {filename}")
   except Exception as e:
        print(f"Error saving the model: {e}") 

def RMSE_ARIMA(data_frame,model_save_path):
    # Division des données en ensembles d'entraînement et de test
    train_size = int(len(data_frame) * 0.8)
    train, test = data_frame['Close'][:train_size], data_frame['Close'][train_size:]

    # Entraîner le modèle ARIMA
    order = (5, 2, 1)
    model = ARIMA(train, order=order)
    print("Avant l'entraînement du modèle")
    fit_model = model.fit()
    print("Apres l'entraînement du modèle") 

    # Sauvegarde du modèle après l'entraînement
    save_arima_model(fit_model,model_save_path)

    # Prédire les valeurs pour l'ensemble de test
    predictions = fit_model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

    # Calculer le RMSE
    rmse_arima = sqrt(mean_squared_error(test, predictions))

    return rmse_arima


try:
   arima_BTC_USD = RMSE_ARIMA(crypto_BTC_USD_df,"ARIMA_BTC_USD.sav")
   arima_BTC_EUR = RMSE_ARIMA(crypto_BTC_EUR_df,"ARIMA_BTC_EUR.sav")
   arima_ETH_USD = RMSE_ARIMA(crypto_ETH_USD_df,"ARIMA_ETH_USD.sav")
   print(f"RMSE: {arima_BTC_USD}")
   print(f"RMSE: {arima_BTC_EUR}")
   print(f"RMSE: {arima_ETH_USD}")
except Exception as e:
  print(f"Error during the main process: {e}")

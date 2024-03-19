#Importation des librairies
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
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
def save_lstm_model(model, filename):
   try:
        pickle.dump(model, open(filename, 'wb'))
        print(f"Model saved successfully as {filename}")
   except Exception as e:
        print(f"Error saving the model: {e}") 


#Fonction relatant de la construction du model lstm
def RMSE_LSTM(data_frame,model_save_path):
 try:    # Extraction de la colonne 'Close' du DataFrame
    data = data_frame.filter(['Close'])

    # Conversion du DataFrame en tableau NumPy
    dataset = data.values

    # Obtention du nombre de lignes pour les données d'entraînement
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Normalisation des données pour les préparer à l'entraînement
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Création du jeu de données d'entraînement
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Construction du modèle LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Avant l'entraînement du modèle")
    # Entraînement du modèle
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
    print("Apres l'entraînement du modèle")
   # Sauvegarde du modèle après l'entraînement
    save_lstm_model(model,model_save_path)

    # Création des données de test
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])


    x_test = np.array(x_test)

    # Reshape des données
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Prédiction avec le modèle LSTM
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calcul du RMSE
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    return rmse
 except Exception as e:
    print(f"Error during training and saving: {e}")
try:
 rmse_BTC_USD = RMSE_LSTM(crypto_BTC_USD_df,"LSTM_BTC_USD.sav")
 rmse_BTC_EUR = RMSE_LSTM(crypto_BTC_EUR_df,"LSTM_BTC_EUR.sav")
 rmse_ETH_USD = RMSE_LSTM(crypto_ETH_USD_df,"LSTM_ETH_USD.sav")
 print(f"RMSE: {rmse_BTC_USD}")
 print(f"RMSE: {rmse_BTC_EUR}")
 print(f"RMSE: {rmse_ETH_USD}")
except Exception as e:
 print(f"Error during the main process: {e}")

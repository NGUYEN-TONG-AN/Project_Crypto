import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import csv
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

#Importation des données à partir du site de Yahoo Finance
ETHUSD = yf.download(tickers=['ETH-USD'], period='Max')
BTCUSD = yf.download(tickers=['BTC-USD'], period='Max')
BTCEUR = yf.download(tickers=['BTC-EUR'], period='Max')

#Modification des index
BTCEUR.reset_index(inplace=True)
BTCEUR = BTCEUR.rename(columns={'index': 'Date'})


ETHUSD.reset_index(inplace=True)
ETHUSD = ETHUSD.rename(columns={'index': 'Date'})

BTCUSD.reset_index(inplace=True)
BTCUSD = BTCUSD.rename(columns={'index': 'Date'})


# Créer la colonne "NP" avec des valeurs de 3
ETHUSD["ID PAIRE"] = 3

# Créer la colonne "Date_NP" qui est la concaténation de "Date" et "NP"
ETHUSD["Date_NP"] = ETHUSD["Date"].astype(str) + ETHUSD["ID PAIRE"].astype(str)
ETHUSD = ETHUSD[['Date_NP'] + [col for col in ETHUSD.columns if col != 'Date_NP']]
ETHUSD = ETHUSD.drop(columns='Adj Close', errors='ignore')
ETHUSD.head(10)

# Créer la colonne "NP" avec des valeurs de 1
BTCUSD["ID PAIRE"] = 1

# Créer la colonne "Date_NP" qui est la concaténation de "Date" et "NP"
BTCUSD["Date_NP"] = BTCUSD["Date"].astype(str) + BTCUSD["ID PAIRE"].astype(str)
BTCUSD = BTCUSD[['Date_NP'] + [col for col in BTCUSD.columns if col != 'Date_NP']]
BTCUSD = BTCUSD.drop(columns='Adj Close', errors='ignore')

BTCUSD.head(10)

# Créer la colonne "NP" avec des valeurs de 2
BTCEUR["ID PAIRE"] = 2

# Créer la colonne "Date_NP" qui est la concaténation de "Date" et "NP"
BTCEUR["Date_NP"] = BTCEUR["Date"].astype(str) + BTCEUR["ID PAIRE"].astype(str)
BTCEUR = BTCEUR[['Date_NP'] + [col for col in BTCEUR.columns if col != 'Date_NP']]
BTCEUR = BTCEUR.drop(columns='Adj Close', errors='ignore')

BTCEUR.head(10)

#Exportation des dataframe au forma CSV
BTCUSD.to_csv("HBTCUSD.csv", index=False)
ETHUSD.to_csv("HETHUSD.csv", index=False)
BTCEUR.to_csv("HBTCEUR.csv", index=False)

csv_files = ["HBTCUSD.csv", "HETHUSD.csv", "HBTCEUR.csv"]

# Étape 1 : Connexion à la base de données
conn = sqlite3.connect("base_de_donnees.db")
cursor = conn.cursor()
# Étape 2 : Création de la table "Donnée historique" si elle n'existe pas

cursor.execute('''CREATE TABLE IF NOT EXISTS "Donnée_historique" (
                  "Date_NP" TEXT PRIMARY KEY,
                   "Date" TEXT,
                  "Open" REAL,
                  "High" REAL,
                  "Low" REAL,
                  "Close" REAL,
                  "Volume" INT,
                  "ID PAIRE" INT,
                  FOREIGN KEY ("ID PAIRE") REFERENCES paire ("ID PAIRE")
               )''')

for file in csv_files:
    # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(file)
    
    # Insérer les données du DataFrame dans la table "Donnée historique"
    df.to_sql("Donnée_historique", conn, if_exists="append", index=False)

conn.commit()
conn.close()

#importer les biliothèque nécessaire pour extrait les donnees sur API BINANCE
import time
import pandas as pd
import os
from binance.client import Client
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
import json
import hmac
import hashlib
import requests
import csv
import os.path
from urllib.parse import urljoin, urlencode
import schedule
import sqlite3
from datetime import datetime

#Fonction cree data base --------------------------------------------------------------------------------------------
def create_database():
    conn = sqlite3.connect('base_de_donnees.db')  # Cree la base de donnes si elle exite pas
    cursor = conn.cursor()
    # Création de la table (si elle n'existe pas)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "Donnees_Streaming" (
            "Date_NP" TEXT PRIMARY KEY,
            "Date" TEXT,
            "Open" REAL,
            "High" REAL,
            "Low" REAL,
            "Close" REAL,
            "Volume" INT,
            "ID PAIRE" INT,
            FOREIGN KEY ("ID PAIRE") REFERENCES paire ("ID PAIRE")
        )
    ''')
    conn.commit()
    conn.close()
#------------------------------------------------------------------------------------------------------------

create_database() # appeller la fonction de creer la data base

#Creer la fonction insert data base--------------------------------------------------------------------------

def insert_csv_data(csv_file_path, symbol):
    if not os.path.isfile(csv_file_path):
        print(f"Le fichier CSV {csv_file_path} n'existe pas. Aucune insertion nécessaire.")
        return
    conn = sqlite3.connect('/home/ubuntu/Project_Crypto/Database/base_de_donnees.db')
    cursor = conn.cursor()

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Ignore the first line if it contains headers
        for row in csv_reader:
            # Generate the current date and time
            now = datetime.now()
            date_heure_extraction = now.strftime("%Y-%m-%d")
            date_np = date_heure_extraction + str(symbol)  # Convertir symbol en une chaîne de caractères et le concaténer
            print(date_np)
            # Adjust the SQL query to match your table structure
            cursor.execute("INSERT INTO 'Donnees_Streaming' (Date_NP, Date, Open, High, Low, Close, Volume, [ID PAIRE]) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (date_np, date_heure_extraction, row[4], row[5], row[6], row[7], row[8], symbol))

    conn.commit()
    conn.close()
#Insérer le fichier CSv dans la base Sql:

insert_csv_data('/home/ubuntu/Project_Crypto/data_extraction/STREAMING_ETH-USD.csv',3)
insert_csv_data('/home/ubuntu/Project_Crypto/data_extraction/STREAMING_BTC-USD.csv',1)
insert_csv_data('/home/ubuntu/Project_Crypto/data_extraction/STREAMING_BTC-EUR.csv',2)

#Supprimer les fichiers Csv une fois l'intégration est réussi:
try:
    os.remove('/home/ubuntu/Project_Crypto/data_extraction/STREAMING_ETH-USD.csv')
    os.remove('/home/ubuntu/Project_Crypto/data_extraction/STREAMING_BTC-USD.csv')
    os.remove('/home/ubuntu/Project_Crypto/data_extraction/STREAMING_BTC-EUR.csv')
    print("Fichiers CSV supprimés avec succès.")
except OSError as e:
    print(f"Erreur lors de la suppression des fichiers CSV : {e}")


#----Cree la table Paire--------------------------------------------------------------------------------------
# Étape 1 : Connexion à la base de données
conn = sqlite3.connect("base_de_donnees.db")
cursor = conn.cursor()

#Création de la table Paire si elle n'existe pas
cursor.execute('''CREATE TABLE IF NOT EXISTS paire (
                  "ID PAIRE" INT PRIMARY KEY,
                  "libellé paire" TEXT
               )''')

# Étape 3 : Insertion des données
cursor.execute(f"SELECT COUNT(*) FROM 'paire'")
count = cursor.fetchone()[0]
if count == 0:
	data = [(1, "BTC/USD"), (2, "BTC/EUR"), (3, "ETH/USD")]
	cursor.executemany("INSERT INTO paire VALUES (?, ?)", data)
cursor.execute(f"SELECT * FROM 'paire'")
# Étape 3 : Récupération des résultats et affichage
rows = cursor.fetchall()


for row in rows:
    print(row)
conn.commit()
conn.close()
#----------------------------------------------------------------------------------------------------------------


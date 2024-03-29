#importer les biliothèque nécessaire pour extrait les donnees sur API BINANCE
#!/usr/bin/python3
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
# !pip install schedule -- pas besoin ce commande dans ubuntu
os.chdir('/home/ubuntu/Project_Crypto/data_extraction')
os.environ['BINANCE_API_KEY_TEST'] = '8Wttc6sFQuJBKFUp0gMBLOj6P4vmITLiiCROFxHRutJFQwNODu4amtw5yRyZOpir'
os.environ['BINANCE_API_SECRET_TEST'] = 'i7RddUrodEuFOtotlWe7vMYg32Xm2UD9FsSirrG1GA62jdQYoc2pJu1zSu17tf2a'

BINANCE_API_KEY_TEST ='8Wttc6sFQuJBKFUp0gMBLOj6P4vmITLiiCROFxHRutJFQwNODu4amtw5yRyZOpir'
BINANCE_API_SECRET_TEST ='i7RddUrodEuFOtotlWe7vMYg32Xm2UD9FsSirrG1GA62jdQYoc2pJu1zSu17tf2a'
#Définir le chemin sur le site Binance

BASE_URL = 'https://api.binance.com'
headers = {
    'X-MBX-APIKEY': BINANCE_API_KEY_TEST
}

class BinanceException(Exception):
    def __init__(self, status_code, data):
        self.status_code = status_code
        if data:
            self.code = data['code']
            self.msg = data['msg']
        else:
            self.code = None
            self.msg = None
        message = f"{status_code} [{self.code}] {self.msg}"
        super().__init__(message)
#---------------------------------------------------------------------------------------------------------------------------
#Définier le Path pour récupérer les valeurs des COINS souhaités

PATH = '/api/v3/ticker'
params1 = {
    'symbol':'ETHUSDT'
}
params2 = {
    'symbol':'BTCUSDT'
}
params3 = {
    'symbol':'BTCEUR'
}
url = urljoin(BASE_URL, PATH)


#Fonction pour automatiser le script extract:
def extract_binance_data():

#Effectuer les requetes pour récupérer les donnes

    r1 = requests.get(url, headers=headers,params=params1)#les donnees récupéré en type JSON
    r2 = requests.get(url, headers=headers,params=params2)#les donnees récupéré en type JSON
    r3 = requests.get(url, headers=headers,params=params3)#les donnees récupéré en type JSON
    r1=r1.json()#transformer en tuple
    r2=r2.json()#transformer en tuple
    r3=r3.json()#transformer en tuple
#print("r1:", r1)  # Afficher le contenu de r1 pour le débogage
# Obtenez le répertoire du script
    #script_directory = os.path.dirname(os.path.abspath(__file__))
   # Spécifiez des chemins absolus pour les fichiers CSV
    #csv_file_path_ethusdt = os.path.join(script_directory, 'STREAMING_ETH-USD.csv')
    #csv_file_path_btcusdt = os.path.join(script_directory, 'STREAMING_BTC-USD.csv')
    #csv_file_path_btceur = os.path.join(script_directory, 'STREAMING_BTC-EUR.csv')
    csv_file_path_ethusdt = '/home/ubuntu/Project_Crypto/data_extraction/STREAMING_ETH-USD.csv' #path pour le fichier CSV ETHUSDT
    csv_file_path_btcusdt = '/home/ubuntu/Project_Crypto/data_extraction/STREAMING_BTC-USD.csv' #path pour le fichier CSV btcusdt
    csv_file_path_btceur = '/home/ubuntu/Project_Crypto/data_extraction/STREAMING_BTC-EUR.csv' #path pour le fichier CSV btceur

#ECRIT LES DONNEES VALEUR ETHUSDT Sur le fichier CSV
    if os.path.isfile(csv_file_path_ethusdt):
        mode = 'a'  # Si le fichier existe, ajoutez les nouvelles lignes
    else:
        mode = 'w'  # Sinon, créez un nouveau fichier

    with open("STREAMING_ETH-USD.csv",mode=mode,newline='') as file:
        writer=csv.writer(file)
        if mode == 'w':  # Si c'est un nouveau fichier, écrivez l'en-tête
        #writer.writerow(['Id''Symbol','Prix_de_fermeture','Date','Variation','prix_moyen','bids','ask',
#'Quantite_achat_par_jour','Quantit_vente_par_jour','Decision'])
            writer.writerow(['symbol','priceChange','priceChangePercent','weightedAvgPrice','openPrice','highPrice','lowPrice',
'lastPrice','volume','quoteVolume','openTime','closeTime','firstId','lastId','count'])
    
        # Extrayez les valeurs nécessaires de l'entrée
        values = [
            r1['symbol'], r1['priceChange'],r1['priceChangePercent'],
            r1['weightedAvgPrice'],r1['openPrice'],r1['highPrice'],
            r1['lowPrice'],r1['lastPrice'],r1['volume'],
            r1['quoteVolume'],r1['openTime'],r1['closeTime'],
            r1['firstId'],r1['lastId'],r1['count']
            ]
        
        writer.writerow(values)
#------------------------------------------------------------------------------------------------------------------------- 
#ECRIT LES DONNEES VALEUR BTCUSDT Sur le fichier CSV

    if os.path.isfile(csv_file_path_btcusdt):
        mode = 'a'  # Si le fichier existe, ajoutez les nouvelles lignes
    else:
        mode = 'w'  # Sinon, créez un nouveau fichier

    with open("STREAMING_BTC-USD.csv",mode=mode,newline='') as file:
        writer=csv.writer(file)
        if mode == 'w':  # Si c'est un nouveau fichier, écrivez l'en-tête
        #writer.writerow(['Id''Symbol','Prix_de_fermeture','Date','Variation','prix_moyen','bids','ask',
#'Quantite_achat_par_jour','Quantit_vente_par_jour','Decision'])
            writer.writerow(['symbol','priceChange','priceChangePercent','weightedAvgPrice','openPrice','highPrice','lowPrice',
'lastPrice','volume','quoteVolume','openTime','closeTime','firstId','lastId','count'])
        values = [
            r2['symbol'], r2['priceChange'],r2['priceChangePercent'],
            r2['weightedAvgPrice'],r2['openPrice'],r2['highPrice'],
            r2['lowPrice'],r2['lastPrice'],r2['volume'],
            r2['quoteVolume'],r2['openTime'],r2['closeTime'],
            r2['firstId'],r2['lastId'],r2['count']
            ]
        writer.writerow(values)

#---------------------------------------------------------------------------------------------------------------------------
#ECRIT LES DONNEES VALEUR BTCEUR Sur le fichier CSV

    if os.path.isfile(csv_file_path_btceur):
        mode = 'a'  # Si le fichier existe, ajoutez les nouvelles lignes
    else:
        mode = 'w'  # Sinon, créez un nouveau fichier

    with open("STREAMING_BTC-EUR.csv",mode=mode,newline='') as file:
        writer=csv.writer(file)
        if mode == 'w':  # Si c'est un nouveau fichier, écrivez l'en-tête
        #writer.writerow(['Id''Symbol','Prix_de_fermeture','Date','Variation','prix_moyen','bids','ask',
#'Quantite_achat_par_jour','Quantit_vente_par_jour','Decision'])
            writer.writerow(['symbol','priceChange','priceChangePercent','weightedAvgPrice','openPrice','highPrice','lowPrice',
'lastPrice','volume','quoteVolume','openTime','closeTime','firstId','lastId','count'])
       
        values = [
            r3['symbol'], r3['priceChange'],r3['priceChangePercent'],
            r3['weightedAvgPrice'],r3['openPrice'],r3['highPrice'],
            r3['lowPrice'],r3['lastPrice'],r3['volume'],
            r3['quoteVolume'],r3['openTime'],r3['closeTime'],
            r3['firstId'],r3['lastId'],r3['count']
            ]
        writer.writerow(values)    
# Planifiez l'exécution de la fonction tous les jours à 18h
#---schedule.every().day.at("18:00").do(extract_binance_data)
extract_binance_data()
# LesCripts suivant est pour l'éxécution automatique hors serveurs Ubuntu sans la commande cron--------------------
#while True:
 #   schedule.run_pending()
    # Vérifiez l'heure actuelle
 #   current_time = time.localtime()
    
    # Si l'heure actuelle est après 18:00, quittez le script
  #  if current_time.tm_hour > 18 or (current_time.tm_hour == 18 and current_time.tm_min >= 0):
   #     break
    
    # Attendez une seconde avant de vérifier à nouveau
   # time.sleep(1)---------------------------------------------------------------------------------------------------

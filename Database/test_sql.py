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
conn = sqlite3.connect("base_de_donnees.db")
cursor = conn.cursor()

# Exécution de la requête pour récupérer les noms des tables
cursor.execute("SELECT * FROM 'Donnees_Streaming'")

rows = cursor.fetchall()

for row in rows:
    print("dat",row)

#cursor.execute("SELECT * FROM 'Donnée historique' WHERE [ID PAIRE] = 3") #---Pour Vérifier les doonees historique
#rows1 = cursor.fetchall()

#for row in rows1:
  # print("dat_historique",row)

conn.commit()
conn.close()

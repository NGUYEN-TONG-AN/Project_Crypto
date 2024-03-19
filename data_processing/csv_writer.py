import sqlite3
import csv
import os
# Connexion à la base de données SQLite
#conn = sqlite3.connect('/home/ubuntu/Project_Crypto/Database/base_de_donnees.db')
#cursor = conn.cursor()

# Exécution de la commande pour renommer la table
#cursor.execute("ALTER TABLE 'Donnees Streaming' RENAME TO 'Donnees_Streaming'")
#cursor.execute("ALTER TABLE 'Donnée historique' RENAME TO 'Donnée_historique'")
# Commit des modifications
#conn.commit()

# Fermeture de la connexion
#conn.close()
def export_table_to_csv(table_name, csv_file_name):
    # Établir la connexion à la base de données SQLite
    conn = sqlite3.connect('/home/ubuntu/Project_Crypto/Database/base_de_donnees.db')
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    existing_tables = cursor.fetchall()

    if existing_tables:
        # Exécuter une requête SQL pour extraire les données de la table spécifiée
        cursor.execute(f"SELECT * FROM {table_name}")

        # Obtenir les en-têtes de colonne à partir de la description de la table
        column_names = [description[0] for description in cursor.description]

        # Vérifier si le fichier CSV existe déjà
        if os.path.isfile(csv_file_name):
            mode = 'a'  # Si le fichier existe, ajoutez les nouvelles lignes
        else:
            mode = 'w'  # Sinon, créez un nouveau fichier

        # Ouvrir le fichier CSV en mode écriture ('w' pour créer ou 'a' pour ajouter) avec l'en-tête
        with open(csv_file_name, mode, newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Si le fichier est nouveau (mode 'w'), écrivez les en-têtes de colonne
            if mode == 'w':
                csv_writer.writerow(column_names)

            # Écrire les données de la table dans le fichier CSV
            for row in cursor.fetchall():
                csv_writer.writerow(row)
    else:
        print(f"Table '{table_name}' does not exist in the database.")

    # Fermer la connexion à la base de données
    conn.close()

# Exporter la première table
export_table_to_csv("Donnees_Streaming", 'Base_Donnees_final.csv')

# Exporter la deuxième table
export_table_to_csv("Donnée_historique", 'Base_Donnees_final.csv')


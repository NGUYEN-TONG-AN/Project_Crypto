import sqlite3
conn = sqlite3.connect("base_utilisateur.db")
cursor = conn.cursor()
cursor.execute("SELECT *  FROM 'users' WHERE username = 'Kani'")

rows = cursor.fetchall()

for row in rows:
    print("dat",row)
conn.commit()
conn.close()

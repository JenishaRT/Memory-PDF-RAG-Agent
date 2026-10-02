import sqlite3

conn = sqlite3.connect("agent.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM structured_documents")
rows = cursor.fetchall()

for row in rows:
    print(row)
    print("---")

conn.close()
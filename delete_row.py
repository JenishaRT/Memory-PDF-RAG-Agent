import sqlite3

conn = sqlite3.connect("agent.db")
conn.execute("DELETE FROM structured_documents")
conn.commit()
conn.close()

print("Row deleted. The agent will now treat that PDF as new.")
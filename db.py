import mysql.connector

def get_connection():

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin123",
        database="communicare_db"
    )

    return connection
import sqlite3
import os

def get_user(conn, username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return conn.execute(query).fetchone()

def read_config(path):
    return eval(open(path).read())

def cleanup(filename):
    os.system("rm -f " + filename)

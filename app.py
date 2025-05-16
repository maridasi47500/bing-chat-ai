from flask import Flask, request
import subprocess
from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

DATABASE = 'mydatabase.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # This enables dict-like access to rows
    return conn

@app.route('/api/job', methods=['GET'])
def get_jobs():
    conn = get_db_connection()
    jobs = conn.execute('SELECT * FROM job').fetchall()
    conn.close()
    return jsonify([dict(row) for row in jobs])

@app.route('/api/sport', methods=['GET'])
def get_sports():
    conn = get_db_connection()
    sports = conn.execute('SELECT * FROM sport').fetchall()
    conn.close()
    return jsonify([dict(row) for row in sports])

@app.route('/api/arts', methods=['GET'])
def get_arts():
    conn = get_db_connection()
    arts = conn.execute('SELECT * FROM arts').fetchall()
    conn.close()
    return jsonify([dict(row) for row in arts])

@app.route('/run_script')
def run_script():
    type_clicked = request.args.get('type')
    if type_clicked == 'job':
        subprocess.run(["python", "your_job_script.py"])
    elif type_clicked == 'art':
        subprocess.run(["python", "your_art_script.py"])
    return '', 204  # No Content

if __name__ == '__main__':
    app.run(debug=True)

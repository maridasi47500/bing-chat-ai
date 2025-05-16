from flask import Flask, request
import subprocess

app = Flask(__name__)

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

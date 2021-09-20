from flask import Flask, jsonify, render_template, request
from markupsafe import escape
import sqlite3

app = Flask(__name__)

# route to root of app. This is what renders viz for users
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

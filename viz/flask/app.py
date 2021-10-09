from flask import Flask, jsonify, render_template, request
from markupsafe import escape
import sqlite3
import pandas as pd

import json
import plotly
import plotly.express as px


app = Flask(__name__)

# route to root of app. This is what renders viz for users
@app.route("/old")
def index():
    return render_template("index.html")

@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return plot_cluster(request.args.get('data'))

@app.route("/")
def cluster_page():
    return render_template('cluster_viz.html', graphJSON=plot_cluster())

def plot_cluster(protein=""):
    cluster_csv_path = "/Users/linda/Downloads/clusters_random_control_clusters.csv"
    df_all = pd.read_csv(cluster_csv_path)
    df = df_all.copy()
    #print(len(df_all))
    df = df.iloc[::100, :]
    if len(protein):
        df = df[df['protein'] == protein]

    print(protein, len(df), len(df_all))

    fig = px.scatter_3d(df, x=df['X'], y=df['Y'], z=df['Z'],
              color=df['protein'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

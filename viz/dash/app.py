import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

cluster_csv_path = "/Users/linda/Downloads/clusters_random_control_clusters.csv"
df = pd.read_csv(cluster_csv_path)

protein_indicators = df['protein'].unique()
cluster_indicators = df['cluster_label'].unique()
table_columns = [c for c in ['protein', 'cluster_label', 'X', 'Y', 'Z'] if c in df.columns]

app.layout = html.Div([
    html.H1('AlphaFold Protein Structural Similarity Explorer'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='protein_filter',
                options=[{'label': i, 'value': i} for i in protein_indicators],
                value='',
                placeholder='Select Target Protein'
            ),
        ],
        style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='cluster_label_filter',
                options=[{'label': i, 'value': i} for i in cluster_indicators],
                value=-1,
                placeholder='Select Cluster ID'
            ),
        ],
        style={'width': '50%', 'display': 'inline-block'}),
    ],
    style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        html.Div([
    	    html.H2('Explorer'),
            dcc.Graph(
                id='cluster-3D-scatter',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ],
        style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
    	    html.H2('Results and Evaluation'),
            dash_table.DataTable(
                id='results-table',
                columns=[{"name": i, "id": i} for i in table_columns],
                data=df.head(12).to_dict('records'),
            ),
        ],
        style={'width': '50%', 'display': 'inline-block',
               'float': 'right', 'padding': '0px 0px 0px 0px'}),
    ],
    style={'width': '100%', 'display': 'inline-block'}),
])

@app.callback(
    Output('cluster-3D-scatter', 'figure'),
    Output('results-table', 'data'),
    Input('protein_filter', 'value'),
        Input('cluster_label_filter', 'value'),
    )
def update_graph(protein, cluster_label):
    #dff = df.iloc[::100, :]
    dff = df
    if protein is not None and len(protein):
        dff = dff[dff['protein'] == protein]
    if cluster_label is not None and cluster_label != -1:
        dff = dff[dff['cluster_label'] == cluster_label]

    fig = px.scatter_3d(dff, x=dff['X'], y=dff['Y'], z=dff['Z'],
              color=dff['cluster_label'], hover_data=['protein'])

    return [fig, dff.head(12).to_dict('records')]

if __name__ == '__main__':
    app.run_server(debug=True)

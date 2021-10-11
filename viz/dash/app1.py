from dash import dash_table
from dash import dcc
from dash import html
from dash import callback_context
from dash import Input, Output
import pandas as pd
import plotly.express as px

from app import app

cluster_csv_path = "/Users/linda/Downloads/clusters_random_control_clusters.csv"
df = pd.read_csv(cluster_csv_path)

protein_indicators = df['protein'].unique()
cluster_indicators = df['cluster_label'].unique()
table_columns = [c for c in ['protein', 'cluster_label', 'X', 'Y', 'Z'] if c in df.columns]

layout = html.Div([
    html.H1('AlphaFold Protein Structural Similarity Explorer'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='protein_filter', className='dropdown',
                options=[{'label': i, 'value': i} for i in protein_indicators],
                value='',
                placeholder='Select Target Protein'
            ),
        ],
        style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='cluster_label_filter', className='dropdown',
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
            html.Div([
            html.Div([
                html.H2('Explorer'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   }),
            html.Div([
                html.Button('Cluster View', n_clicks=0, 
                    id='cluster-view-button', className='view-button',
                    disabled=True),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '25px 0px'}),
            html.Div([
                html.Button('Confidence View', n_clicks=0,
                    id='confidence-view-button', className='view_button'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '25px 0px'}),
            ]),
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

def toggle_view():
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    cluster_view = True
    confidence_view = False

    if 'cluster-view-button' in changed_id:
        cluster_view = True
        confidence_view = False
    elif 'confidence-view-button' in changed_id:
        cluster_view = False
        confidence_view = True
    else:
        pass
    return cluster_view, confidence_view

@app.callback(
    Output('cluster-3D-scatter', 'figure'),
    Output('results-table', 'data'),
    Output('cluster-view-button', 'disabled'),
    Output('confidence-view-button', 'disabled'),
    Input('protein_filter', 'value'),
    Input('cluster_label_filter', 'value'),
    Input('cluster-view-button', 'n_clicks'),
    Input('confidence-view-button', 'n_clicks'),
    )
def update_graph(protein, cluster_label, 
            cluster_view_button, confidence_view_button):
    # Handle View Buttons
    cluster_view, confidence_view = toggle_view()

    #dff = df.iloc[::100, :]
    dff = df
    if protein is not None and len(protein):
        dff = dff[dff['protein'] == protein]
    if cluster_label is not None and cluster_label != -1:
        dff = dff[dff['cluster_label'] == cluster_label]

    color = dff['cluster_label']
    if cluster_view:
        color = dff['cluster_label']
    elif confidence_view:
        # TODO Set to confidence value!
        color = dff['X']

    fig = px.scatter_3d(dff, x=dff['X'], y=dff['Y'], z=dff['Z'],
              color=color, hover_data=['protein'])

    return [fig, dff.head(12).to_dict('records'),
            cluster_view, 
            confidence_view,
           ]


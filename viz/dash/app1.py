from dash import dash_table
from dash import dcc
from dash import html
from dash import callback_context
from dash import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import pickle
import numpy as np


from app import app

use_random_baseline = False

if use_random_baseline:
    cluster_csv_path = "/Users/linda/Downloads/clusters_random_control_clusters.csv"
    df = pd.read_csv(cluster_csv_path)
    df = df.rename(columns={'cluster_label': 'Cluster Label'})
    cluster_df = pd.read_csv(cluster_csv_path)
    cluster_df = df.rename(columns={'cluster_label': 'Cluster Label'})
else:
    cluster_parquet_path = "/Users/linda/Downloads/all_proteins_with_confidence.parquet"
    df = pd.read_parquet(cluster_parquet_path)

    cluster_stats_path = "/Users/linda/Downloads/samples_cluster_stats2.parquet"
    cluster_df = pd.read_parquet(cluster_stats_path)
    cluster_df = cluster_df.reset_index()
    cluster_df = cluster_df.rename(columns={'cluster_label': 'Cluster Label'})

    table_path = "/Users/linda/Downloads/pairwise_proteins2.parquet"
    table_df = pd.read_parquet(table_path)
    table_df = table_df.rename(columns={'target_protein': 'result_protein', 'query_protein': 'target_protein'})

    model_path = "/Users/linda/Downloads/mvp_input_2021_1029_0054_model_overview.pkl"
    with open(model_path, 'rb') as f:
	    model_dict = pickle.load(f)
    model_dict['Median number of proteins per cluster'] = cluster_df['count'].median()
    model_dict['Total number of proteins'] = cluster_df['count'].sum()


#cluster_df = cluster_df.sample(1000)
#df = df.sample(1000)
#table_df = table_df.sample(1000)

df = df.astype({'Cluster Label': 'int32'})

protein_indicators = df['protein'].unique()
cluster_indicators = df['Cluster Label'].unique()
table_columns = [c for c in ['target_protein', 'result_protein', 'aligned_length', 'bitscore', 'evalue', 'rmsd', 'tmalign_score'] if c in table_df.columns]
#table_columns = [c for c in ['target_protein', 'result_protein', 'aligned_length', 'rmsd', 'tmalign_score'] if c in table_df.columns]
#table_columns = [c for c in ['aligned_length', 'rmsd', 'tmalign_score'] if c in table_df.columns]

all_colors = px.colors.qualitative.Plotly
num_colors = len(all_colors)
color_discrete_map = {str(i): all_colors[idx % num_colors] \
        for idx, i in enumerate(cluster_indicators)}

# Store last view selection
global last_views
last_views = None


layout = html.Div([
    html.H1('AlphaFold2 Protein Structural Similarity Explorer'),
	html.Div([
        html.H6('Model Summary', style={'display': 'inline', 'margin-right': '10px'}),
        html.Div(children= '%s: %s' % ('Total Proteins', model_dict['Total number of proteins']),
            id='model-noise', className='modelstats'),
		html.Div(children= '%s: %s' % ('Total Clusters', model_dict['Number of clusters (excl. noise)']),
            id='model-clusters', className='modelstats'),
        html.Div(children= '%s: %2.2f%%' % ('Percent of Proteins Unclustered' ,model_dict['Noise as % of total'] * 100),
            id='model-noise-percent', className='modelstats'),
        html.Div(children= '%s: %s Proteins' % ('Largest Cluster' ,model_dict['Largest non-noise cluster']),
            id='model-non-noise', className='modelstats'),
        html.Div(children= '%s: %d Proteins' % ('Median Cluster Size', model_dict['Median number of proteins per cluster']),
            id='model-median', className='modelstats'),
    ],
	style={'width': '100%', 'display': 'inline-block', 'padding': '0px 0px 15px 0px', }),
    #html.H5('Query by Protein or Cluster'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='protein_filter', className='dropdown',
                options=[{'label': i, 'value': i} for i in protein_indicators],
                value='',
                placeholder='Select Target Protein',
                multi=True,
            ),
        ],
        style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='cluster_label_filter', className='dropdown',
                options=[{'label': i, 'value': i} for i in cluster_indicators],
                value=-999,
                placeholder='Select Cluster ID',
                multi=True,
            ),
        ],
        style={'width': '50%', 'display': 'inline-block'}),
    ],
    style={'width': '50%', 'display': 'inline-block', 'padding': '0px 0px 0px 0px'}),


        html.Div([
                #html.H5('Cluster Comparisons'),
			html.Div([
				html.H6('Cluster Explorer'),
            	dcc.Graph(
                	id='cluster-num-hist',
				)
			],
		    className='panel',
        	style={'width': '48%', 'display': 'inline-block'}),
		
			html.Div([
				html.H6('Protein Sequence Lengths Per Cluster'),
            	dcc.Graph(
                	id='cluster-protein-length',
				)
			],
		    className='panel',
        	style={'width': '48%', 'display': 'inline-block',
               'float': 'right',}),
		],
		style={'width': '100%', 'display': 'inline-block', 'padding': '0px 0px 10px 0px'}),


    html.Div([
        #html.H5('Protein Comparisons'),
        html.Div([
            html.Div([
            html.Div([
                html.H6('Protein Explorer'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   }),
            html.Div([
                html.Button('Cluster View', n_clicks=0, 
                    id='cluster-view-button', className='view-button',
                    disabled=True),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '12px 0px'}),
            html.Div([
                html.Button('Functional View', n_clicks=0,
                    id='functional-view-button', className='view_button'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '12px 0px'}),
            html.Div([
                html.Button('Confidence View', n_clicks=0,
                    id='confidence-view-button', className='view_button'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '12px 0px'}),
            ]),
            dcc.Graph(
                id='cluster-3D-scatter',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            ),
        ],
		className='panel',
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.H6('Protein Results and Evaluation'),
            dash_table.DataTable(
                id='results-table',
                columns=[{"name": i, "id": i} for i in table_columns],
                data=table_df.head(100).to_dict('records'),
				sort_action='native',
				filter_action='native',
				page_action="native",
        		page_current=0,
        		page_size=10,
            ),
        ],
		className='panel',
        style={'width': '48%', 'display': 'inline-block',
               'float': 'right',}),
    ],
    style={'width': '100%', 'display': 'inline-block', 'padding': '0px 0px 10px 0px'}),

])

def toggle_view():
    global last_views

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    cluster_view = True
    confidence_view = False
    functional_view = False

    if 'cluster-view-button' in changed_id:
        cluster_view = True
        confidence_view = False
        functional_view = False
    elif 'confidence-view-button' in changed_id:
        cluster_view = False
        confidence_view = True
        functional_view = False
    elif 'functional-view-button' in changed_id:
        cluster_view = False
        confidence_view = False
        functional_view = True
    elif last_views:
        cluster_view, confidence_view, functional_view = last_views
    last_views = [cluster_view, confidence_view, functional_view]
    return cluster_view, confidence_view, functional_view

@app.callback(
    Output('protein_filter', 'disabled'),
    Output('cluster_label_filter', 'disabled'),
    Output('cluster-3D-scatter', 'figure'),
    Output('results-table', 'data'),
    Output('cluster-num-hist', 'figure'),
    Output('cluster-protein-length', 'figure'),
    Output('cluster-view-button', 'disabled'),
    Output('confidence-view-button', 'disabled'),
    Output('functional-view-button', 'disabled'),
    Output('cluster_label_filter', 'value'),
	Output('cluster-num-hist', 'clickData'),
    Input('protein_filter', 'value'),
    Input('cluster_label_filter', 'value'),
    Input('cluster-view-button', 'n_clicks'),
    Input('confidence-view-button', 'n_clicks'),
    Input('functional-view-button', 'n_clicks'),
	Input('cluster-3D-scatter', 'clickData'),
	Input('cluster-num-hist', 'clickData'),
    )
def update_graph(protein, cluster_label, 
            cluster_view_button, confidence_view_button, functional_view_button,
            protein_click_data, cluster_click_data):
    link = 'https://www.uniprot.org/uniprot/'
    if protein_click_data:
        link += protein_click_data['points'][0]['customdata'][0]
        webbrowser.open(link)

    if cluster_click_data:
        cluster_label = [cluster_click_data['points'][0]['customdata']]
        cluster_click_data = None

    # Handle View Buttons
    cluster_view, confidence_view, functional_view = toggle_view()

    #dff = df.iloc[::100, :]
    dff = df
    cluster_dff = cluster_df
    table_dff = table_df


    disable_protein_filter = False
    disable_cluster_filter = False
    if protein is not None and len(protein):
        cluster_id = dff[dff['protein'].isin(protein)]['Cluster Label'].unique()
        dff = dff[dff['Cluster Label'].isin(cluster_id)]
        table_dff = table_dff[table_dff['target_protein'].isin(protein) & table_dff['result_protein'].isin(dff['protein'].unique())]
        #table_dff = table_dff[table_dff['target_protein'].isin(protein)]
        disable_cluster_filter = True
    if cluster_label is not None and isinstance(cluster_label, list) and len(cluster_label):
        dff = dff[dff['Cluster Label'].isin(cluster_label)]
        disable_protein_filter = True
    cluster_dff = cluster_dff[cluster_dff['Cluster Label'].isin(dff['Cluster Label'].unique())]


    # Adjust dropdown menus
    """
    protein_options = []
    unique_proteins = dff['protein'].unique()
    for i in protein_indicators:
        if i not in unique_proteins:
            protein_options.append({'label': i, 'value': i, 'disabled': True})
        else:
            protein_options.append({'label': i, 'value': i, 'disabled': False})

    unique_p
    cluster_options = [{'label': i, 'value': i} for i in cluster_indicators]
    """


    scatter_dff = dff #dff.sample(1000)
    color = scatter_dff['Cluster Label']
    if cluster_view:
        color = scatter_dff['Cluster Label'].astype(str)
    elif confidence_view:
        # TODO Set to confidence value!
        color = scatter_dff['confidence']
    elif functional_view:
        # TODO Set to functional value!
        color = scatter_dff['Y']

    scatter_fig = px.scatter_3d(scatter_dff, x=scatter_dff['X'], y=scatter_dff['Y'], z=scatter_dff['Z'],
              color=color, hover_data={'protein': True, 'length': True, 'Cluster Label': True}, color_discrete_map=color_discrete_map, labels={'color': 'Cluster', 'confidence': 'Confidence'})
    scatter_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=True)

    """
    cluster_groups = dff.groupby(['Cluster Label'])

    points_count = cluster_groups.count()
    points_count = points_count.rename(columns={'protein': 'Count'})
    points_count[['X', 'Y', 'Z']] = cluster_groups[['X', 'Y', 'Z']].mean()
    points_count.reset_index(inplace=True)
    """

    """
    cluster_dfff = cluster_dff[cluster_dff['count'] < 100]
    num_hist = px.histogram(
	    x=cluster_dfff['count'],
	    color=cluster_dfff['Cluster Label'].astype(str),
        color_discrete_map=color_discrete_map,
		#log_x=True,
		nbins = 100,
        labels ={'color': 'cluster'},
        )
    """

    cluster_groups = dff.groupby(['Cluster Label'])
    points_count = cluster_groups.count()
    points_count[['X', 'Y', 'Z']] = cluster_groups[['X', 'Y', 'Z']].mean()
    points_count = points_count.reset_index()
    df3 = points_count
    df3['log_protein'] = df3['protein'].apply(lambda x: np.log(x))

    """
    num_hist = px.scatter_3d(df3, x=df3['X'], y=df3['Y'], z=df3['Z'], size=df3['log_protein'],
              color=df3['Cluster Label'].astype(str), color_discrete_map=color_discrete_map)
	"""

    num_hist = go.Figure(data=go.Scatter3d(
        x=df3['X'],
        y=df3['Y'],
        z=df3['Z'],
		customdata=df3['Cluster Label'],
        text=df3['Cluster Label'].astype(str),
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=0.2,
            size=df3['log_protein'],
            color=df3['Cluster Label'].apply(lambda x: color_discrete_map[str(x)]),
            #colorscale = 'sunsetdark',
            #colorbar_title = 'Log(Number of Proteins)',
            line_color='rgba(255, 255, 255, 0)'
        )
    ))
    num_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend = False)


    """
	num_hist = px.scatter_3d(points_count, 
	          x=points_count['X'], y=points_count['Y'], z=points_count['Z'],
              color=points_count['Cluster Label'], hover_data=['Cluster Label'],
			  size=points_count['Count'] / points_count['Count'].max())
    """

    # TODO Set to Protein Sequence Length!
    cluster_dfff = cluster_dff[cluster_dff['median_seq_len'] < 2500]
    length_hist = px.histogram(
	    x=cluster_dfff['median_seq_len'],
	    color=cluster_dfff['Cluster Label'].astype(str),
        color_discrete_map=color_discrete_map,
		#log_x=True,
		nbins = 100,
        labels ={'color': 'Cluster'},
       )
    length_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Model level data

    return [
            disable_protein_filter,
            disable_cluster_filter,
	        scatter_fig, 
	        table_dff.to_dict('records'),
			num_hist,
			length_hist,
            cluster_view, 
            confidence_view,
			functional_view,
			cluster_label,
			cluster_click_data,
           ]

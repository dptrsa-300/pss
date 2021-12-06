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

#import dash_bootstrap_components as dbc

#from time import time

from app import app

#from flask_caching import Cache
import os

"""
cache = Cache(app.server, config={
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache',
    #'CACHE_TYPE': 'redis',
    #'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')
})
"""


#start = time()

pd.options.display.float_format = '${:.2f}'.format

use_random_baseline = False
subsamples = None #2000

if use_random_baseline:
    cluster_csv_path = "assets/clusters_random_control_clusters.csv"
    df = pd.read_csv(cluster_csv_path)
    df = df.rename(columns={'cluster_label': 'Cluster Label'})
    cluster_df = pd.read_csv(cluster_csv_path)
    cluster_df = df.rename(columns={'cluster_label': 'Cluster Label'})
else:
    cluster_parquet_path = "assets/all_proteins_with_confidence.parquet"
    df = pd.read_parquet(cluster_parquet_path)

    cluster_stats_path = "assets/samples_cluster_stats2.parquet"
    cluster_df = pd.read_parquet(cluster_stats_path)
    cluster_df = cluster_df.reset_index()
    cluster_df = cluster_df.rename(columns={'cluster_label': 'Cluster Label'})

    table_path = "assets/pairwise_proteins2.parquet"
    table_df = pd.read_parquet(table_path)
    table_df = table_df.rename(columns={'target_protein': 'result_protein', 'query_protein': 'target_protein', 'cluster': 'Cluster Label'})

    model_path = "assets/model_overview.pkl"
    with open(model_path, 'rb') as f:
	    model_dict = pickle.load(f)
    model_dict['Median number of proteins per cluster'] = cluster_df['count'].median()
    model_dict['Total number of proteins'] = cluster_df['count'].sum()

if subsamples:
    #cluster_df = cluster_df.sample(min(subsamples, len(cluster_df))).sort_values(by=['Cluster Label'])
    df = df.sample(min(subsamples, len(df))).sort_values(by=['Cluster Label'])
    cluster_df = cluster_df[cluster_df['Cluster Label'].isin(df['Cluster Label'].unique())].sort_values(by=['Cluster Label'])
    table_df = table_df.sample(min(subsamples, len(table_df))).sort_values(by=['target_protein'])

df = df.astype({'Cluster Label': 'int32'})

def compute_closest_clusters(df):
    cluster_groups = df.groupby(['Cluster Label'])
    cluster_df = cluster_groups.count().reset_index()
    cluster_xyz = cluster_groups[['X', 'Y', 'Z']].mean().to_numpy()
    cluster_label = cluster_df['Cluster Label']
    distances = np.linalg.norm(cluster_xyz[:, np.newaxis] - cluster_xyz[np.newaxis, :], axis=-1)
    closest_clusters = {label: cluster_label.iloc[np.argsort(dist)[1:].tolist()].to_list() for label, dist in zip(cluster_label, distances)}
    return closest_clusters

#closest_clusters = compute_closest_clusters(df)

protein_indicators = df['protein'].unique()
cluster_indicators = df['Cluster Label'].unique()
table_columns = [c for c in ['target_protein', 'result_protein', 'aligned_length', 'bitscore', 'evalue', 'rmsd', 'tmalign_score'] if c in table_df.columns]
#table_columns = [c for c in ['target_protein', 'result_protein', 'aligned_length', 'rmsd', 'tmalign_score'] if c in table_df.columns]
#table_columns = [c for c in ['aligned_length', 'rmsd', 'tmalign_score'] if c in table_df.columns]

for c in ['bitscore', 'rmsd', 'tmalign_score']:
    table_df[c]=table_df[c].map('{:.2f}'.format)
table_df['evalue']=table_df['evalue'].map('{:.2e}'.format)
table_df['aligned_length']=table_df['aligned_length'].map('{:.0f}'.format)
table_records = table_df.to_dict('records')

#protein2cluster = {row['protein']: row['Cluster Label'] for index, row in df.iterrows()}
cluster_table_df = table_df.dropna()
cluster_table_df['tmalign_score'] = cluster_table_df['tmalign_score'].astype(float)
cluster_table_df['rmsd'] = cluster_table_df['rmsd'].astype(float)
#cluster_table_df['Target Cluster Label'] = cluster_table_df['target_protein'].apply(
#        lambda x: protein2cluster[x])
#cluster_table_df['Result Cluster Label'] = cluster_table_df['result_protein'].apply(
#        lambda x: protein2cluster[x])
cluster_table_df = cluster_table_df.groupby(['Cluster Label']).mean().reset_index()

all_colors = px.colors.qualitative.Plotly
num_colors = len(all_colors)
color_discrete_map = {str(i): all_colors[idx % num_colors] \
        for idx, i in enumerate(cluster_indicators)}

# Store last view selection
global last_views
last_views = None

#print(time() - start, 'init')

layout = html.Div([
    #dbc.Spinner(color="primary"),
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
        html.P(
            [
                'Instructions:',
                html.Br(),
                '(A) Start here! Tell us your protein(s) of interest - type or select from the dropdown. We’ll show you the cluster(s) your proteins belong to.',
                html.Br(),
                '(B) (Advanced Users Only) Instead of target proteins, you can also select target cluster(s) (if you already know which cluster(s) you’d like).',
                html.Br(),
                '(C) You can also tell us how many additional neighboring clusters you’d like to explore.',
                html.Br(),
                'We recommend only using A + C for first-time users.'
            ],
            className='explanation'),
        html.Div([
            dcc.Dropdown(
                id='protein_filter', className='dropdown',
                options=[{'label': i, 'value': i} for i in protein_indicators],
                value='',
                placeholder='(A) Type or Select Target Protein',
                multi=True,
            ),
        ],
        style={'width': '33.333333%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='cluster_label_filter', className='dropdown',
                options=[{'label': i, 'value': i} for i in cluster_indicators],
                value=-999,
                placeholder='(B) Type or Select Cluster ID',
                multi=True,
            ),
        ],
        style={'width': '33.333333%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='cluster_count_filter', className='dropdown',
                options=(lambda x: x[0].update({'label': '1 Cluster'}) or x)([{'label': str(i)  + ' Clusters', 'value': i} for i in range(1, len(cluster_indicators))]),
                value=0,
                placeholder='(C) Select Number of Next Closest Clusters to Show',
				disabled=True,
                multi=False,
            ),
        ],
        style={'width': '33.333333%', 'display': 'inline-block'}),
    ],
    style={'width': '100%', 'display': 'inline-block', 'padding': '0px 0px 0px 0px'}),


        html.Div([
                #html.H5('Cluster Comparisons'),
			html.Div([
				html.H6('Cluster Explorer'),
                html.P(
                    'Each bubble in this 3D space represents a cluster. Pan, zoom, spin - hover for annotations, click to see details (all charts will update).', 
                    className='explanation'),
            	dcc.Graph(
                	id='cluster-num-hist',
				)
			],
		    className='panel',
        	style={'width': '48%', 'display': 'inline-block'}),
		
			html.Div([
				#html.H6('Protein Sequence Lengths Per Cluster'),
				html.H6('Industry / Gold Standard Structural Similarity Metrics'),
                html.P(
                    'Each cluster’s average TM-Align and RMSD scores are displayed. Strong similarity indicates strong confidence in protein structural similarity within a cluster.',
                    className='explanation'),
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
                    id='cluster-view-button', className='view_button',
                    disabled=True),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '12px 0px'}),
            html.Div([
                html.Button('Functional View', n_clicks=0,
                    id='functional-view-button', className='view_button'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '12px 0px', 'display': 'none'}),
            html.Div([
                html.Button('Confidence View', n_clicks=0,
                    id='confidence-view-button', className='view_button'),
            ],
	    style={'width': '25%', 'display': 'inline-block', 
                   'float': 'right', 'padding': '12px 0px'}),
            ]),
            html.Div([
                html.P(
                    'Each point in this 2D space represents a protein. Pan, zoom - hover for annotations, click to see Uniprot details (opens in new tab).',
                    className='explanation'),
            ],
	    style={'width': '100%', 'display': 'inline-block', 
                   'float': 'none', 'padding': '0px 0px'}),
            dcc.Graph(
                id='cluster-3D-scatter',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            ),
        ],
		className='panel',
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.H6('Protein Results and Evaluation'),
            html.P([
                'Each row in this table represents a protein pair.',
                html.Br(),
                'Sort or filter by any column.'
                ],
                className='explanation'),
            dash_table.DataTable(
                id='results-table',
                columns=[{"name": i, "id": i} for i in table_columns],
                data=table_records,
				sort_action='native',
				filter_action='native',
				page_action="native",
        		page_current=0,
        		page_size=10,
                style_data={'backgroundColor': 'transparent'},
                style_header={'backgroundColor': 'transparent'},
                style_filter={'backgroundColor': 'transparent', 'color': 'white'},
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

app.clientside_callback(
    """
    function(protein_click_data) {
        link = 'https://www.uniprot.org/uniprot/'
        if (protein_click_data) {
            link += protein_click_data["points"][0]["customdata"][0]
            window.open(link)
        }
        return protein_click_data
    }
    """,
    Output('cluster-3D-scatter', 'clickData'),
    Input('cluster-3D-scatter', 'clickData'),
    )

@app.callback(
    Output('protein_filter', 'disabled'),
    Output('cluster_label_filter', 'disabled'),
    Output('cluster_count_filter', 'disabled'),
    Output('cluster-3D-scatter', 'figure'),
    #Output('results-table', 'data'),
    Output('cluster-num-hist', 'figure'),
    Output('cluster-protein-length', 'figure'),
    Output('cluster-view-button', 'disabled'),
    Output('confidence-view-button', 'disabled'),
    Output('functional-view-button', 'disabled'),
    Output('cluster_label_filter', 'value'),
	Output('cluster-num-hist', 'clickData'),
    Input('protein_filter', 'value'),
    Input('cluster_label_filter', 'value'),
    Input('cluster_count_filter', 'value'),
    Input('cluster-view-button', 'n_clicks'),
    Input('confidence-view-button', 'n_clicks'),
    Input('functional-view-button', 'n_clicks'),
	#Input('cluster-3D-scatter', 'clickData'),
	Input('cluster-num-hist', 'clickData'),
    )
#@cache.memoize(timeout=60)
def update_graph(protein, cluster_label, num_neighbor_clusters,
            cluster_view_button, confidence_view_button, functional_view_button,
            #protein_click_data, 
            cluster_click_data):
    #start = time()
    if cluster_click_data:
        cluster_label = [cluster_click_data['points'][0]['customdata']]
        cluster_click_data = None

    # Handle View Buttons
    cluster_view, confidence_view, functional_view = toggle_view()

    #dff = df.iloc[::100, :]
    dff = df
    cluster_dff = cluster_df
    table_dff = table_df

    #print(time() - start, "1")
    #start = time()


    closest_clusters = compute_closest_clusters(dff)
    def add_neighboring_clusters(cluster_label, num_neighbors):
        input_cluster_label = cluster_label[:]
        for cluster in input_cluster_label:
            cluster_label += closest_clusters[cluster][:num_neighbor_clusters]
        cluster_label = np.unique(cluster_label).tolist()
        return cluster_label


    disable_protein_filter = False
    disable_cluster_filter = False
    disable_count_filter = True
    if protein is not None and len(protein):
        cluster_id = list(dff[dff['protein'].isin(protein)]['Cluster Label'].unique())
        if num_neighbor_clusters is not None and num_neighbor_clusters > 0:
            cluster_id = add_neighboring_clusters(cluster_id, num_neighbor_clusters)
        dff = dff[dff['Cluster Label'].isin(cluster_id)]

        table_dff = table_dff[table_dff['target_protein'].isin(protein) & table_dff['result_protein'].isin(dff['protein'].unique())]
        #table_dff = table_dff[table_dff['target_protein'].isin(protein)]

        disable_cluster_filter = True
        disable_count_filter = False

    if cluster_label is not None and isinstance(cluster_label, list) and len(cluster_label):
        cluster_id = cluster_label[:]
        if num_neighbor_clusters is not None and num_neighbor_clusters > 0:
            cluster_id = add_neighboring_clusters(cluster_id, num_neighbor_clusters)
        dff = dff[dff['Cluster Label'].isin(cluster_id)]

        disable_protein_filter = True
        disable_count_filter = False

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
    #print(len(scatter_dff), 'before')
    scatter_dff = scatter_dff[scatter_dff['Cluster Label'] != -1]
    #print(len(scatter_dff), 'after')
    #scatter_dff = scatter_dff.sample(min(1000, len(scatter_dff))).sort_values(by=['Cluster Label'])
    scatter_dff['Cluster Label'] = scatter_dff['Cluster Label'].astype(str)
    if cluster_view:
        color = 'Cluster Label'
    elif confidence_view:
        # TODO Set to confidence value!
        color = 'confidence'
    elif functional_view:
        # TODO Set to functional value!
        color = 'Y'
    else:
        color = 'Cluster Label'

    #print(time() - start, "2")
    #start = time()

    scatter_fig = px.scatter(scatter_dff, x='X', y='Y',
            #z='Z',
              color=color,
              hover_data={'protein': True, 'length': True, 'Cluster Label': True,
                  'X': False, 'Y': False},
              color_discrete_map=color_discrete_map,
              labels={'color': 'Cluster', 'confidence': 'Confidence', 'protein': 'Protein',
                  'length': 'Length',
                  },
              render_mode='webgl',
              #template="plotly_dark",
              )
    scatter_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        font_color = "#FFFFFF",
	    #scene = dict(
        yaxis={'visible': True, 'showticklabels': False, 'title': ''},
        xaxis={'visible': True, 'showticklabels': False, 'title': ''},
        #zaxis={'visible': True, 'showticklabels': False, 'title': ''},
        #),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    #print(time() - start, "3")
    #start = time()

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
    df3['log_protein'] = df3['protein'].apply(lambda x: np.log(x) + 1.0)

    """
    num_hist = px.scatter_3d(df3, x=df3['X'], y=df3['Y'], z=df3['Z'], size=df3['log_protein'],
              color=df3['Cluster Label'].astype(str), color_discrete_map=color_discrete_map)
	"""
    #print(len(df3), 'before3 cluster')

    df3 = df3.set_index('Cluster Label').join(cluster_dff.set_index('Cluster Label')).reset_index()

    def create_text(row):
        text = ""
        text += "Cluster Label=" + str(int(row['Cluster Label']))
        text += "<br>" + "Protein Count=" + str(int(row['protein']))
        text += "<br>" + "AlphaFold2 pLDDT=" + "%.2f" % row['protein_confidence'] + "%"
        text += "<br>" + "Mean Sequence Length=" + "%.2f" % row['mean_seq_len']
        text += "<br>" + "StdDev Sequence Length=" + "%.2f" % row['std_seq_len']
        text += "<br>" + "%.2f" % (np.nan_to_num(row['top_go_occurrence']) * 100) + \
                "% of Cluster " + \
                "Belongs to=<br>Functional Group " +  str(row['top_go_id']) + "<br>(" + \
                str(row['top_go_name']) + ")"
        #text += "<br>" + "Top Go Count: " + "%d" % np.nan_to_num(row['top_go_count'])
        #text += "<br>" + "Top Go Name: " + str(row['top_go_name'])
        #text += "<br>" + "Top Go : " + "%.2f" % np.nan_to_num(row['top_go_occurrence'])
        return text

    df3['Text'] = df3.apply(create_text, axis="columns")

    hovertemplate = "<br>".join([
        #"X=%{x}",
        #"Y=%{y}",
        #"Z=%{z}",
        "%{text}",
    ]) + "<extra></extra>"

    num_hist = go.Figure(data=go.Scatter3d(
        x=df3['X'],
        y=df3['Y'],
        z=df3['Z'],
        customdata=df3['Cluster Label'],
        text=df3['Text'].astype(str),
        hovertemplate=hovertemplate,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=0.25,
			#sizemin=4,
            size=df3['log_protein'],
            color=df3['Cluster Label'].apply(lambda x: color_discrete_map[str(x)]),
            #colorscale = 'sunsetdark',
            #colorbar_title = 'Log(Number of Proteins)',
            line_color='rgba(255, 255, 255, 0)'
        )
    ))

    #print(time() - start, "4")
    #start = time()

    num_hist.update_layout(
        font_color = "#FFFFFF",
        scene_camera = {'eye': dict(x=1.0, y=1.0, z=0.5)},
        margin=dict(l=0, r=0, t=0, b=0), showlegend = False,
	    scene = dict(
            yaxis={'visible': True, 'showticklabels': False,
                'title': '', 'backgroundcolor': 'rgba(0,0,0,0)'},
            xaxis={'visible': True, 'showticklabels': False,
                'title': '', 'backgroundcolor': 'rgba(0,0,0,0)'},
            zaxis={'visible': True, 'showticklabels': False,
                'title': '', 'backgroundcolor': 'rgba(0,0,0,0)'},
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )


    """
	num_hist = px.scatter_3d(points_count, 
	          x=points_count['X'], y=points_count['Y'], z=points_count['Z'],
              color=points_count['Cluster Label'], hover_data=['Cluster Label'],
			  size=points_count['Count'] / points_count['Count'].max())
    """

    """
    cluster_dfff = cluster_dff[cluster_dff['median_seq_len'] < 2500]
    length_hist = px.histogram(
	    x=cluster_dfff['median_seq_len'],
	    color=cluster_dfff['Cluster Label'].astype(str),
        color_discrete_map=color_discrete_map,
		#log_x=True,
		nbins = 100,
        labels ={'color': 'Cluster'},
       )
    length_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis={'title': 'sequence length'})
    """

    #print(time() - start, "5")
    #start = time()


    """
    table_dfff = table_df.dropna()
    left = dff.set_index('protein')
    right = table_dfff.set_index('target_protein')
    new = left.join(right)
    table_dfff = new.reset_index()
    table_dfff = table_dfff.rename(columns={'index': 'target_protein'})
    table_dfff = table_dfff.dropna()
    print(len(table_dfff), 'before2')
    #table_dfff = table_dfff[table_dfff['Cluster Label'] != -1]
    #table_dfff = table_dfff.groupby(['Cluster Label']).mean()
    table_dfff = table_dfff.sample(min(1000, len(table_dfff))).sort_values(by=['Cluster Label'])
    print(len(table_dfff), 'after2')
    table_dfff['tmalign_score'] = table_dfff['tmalign_score'].astype(float)
    table_dfff['rmsd'] = table_dfff['rmsd'].astype(float)
    length_hist = px.scatter(table_dfff,
	    x='tmalign_score',
		y='rmsd',
		color=table_dfff['Cluster Label'].astype(str),
        #color_discrete_map=color_discrete_map, 
        hover_data={'target_protein': True, 'result_protein': True},
	    #color=table_dff['Cluster Label'].astype(str),
        color_discrete_map=color_discrete_map,
		#log_x=True,
        labels ={'color': 'Cluster'},
        render_mode='webgl'
       )
    length_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis={'title': 'TM-Align Score'}, yaxis={'title': 'RMSD'})
    """

    table_dfff = cluster_table_df[
            cluster_table_df['Cluster Label'].isin(dff['Cluster Label'].unique())]
    length_hist = px.scatter(table_dfff,
	    x='tmalign_score',
		y='rmsd',
		color=table_dfff['Cluster Label'].astype(str),
        #hover_data={'target_protein': True, 'result_protein': True},
	    #color=table_dff['Cluster Label'].astype(str),
        color_discrete_map=color_discrete_map,
		#log_x=True,
        labels ={'color': 'Cluster Label', 'rmsd': 'RMSD', 'tmalign_score': 'TM-Align Score'},
        render_mode='webgl'
       )
    length_hist.update_layout(
        font_color = "#FFFFFF",
        margin=dict(l=0, r=0, t=0, b=0), 
        xaxis={'title': 'TM-Align Score', 'range': [0.0, 1.0]}, 
        yaxis={'title': 'RMSD', 'range': [0.0, 12.0]},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
            )
    length_hist.add_trace(go.Scatter(
        x = [0.5, 0.5, 1.0],
        y = [0.0, 3.0, 3.0],
        line=dict(color="#FFFFFF", width=4),
        showlegend=False,
        ))
    length_hist.add_trace(go.Scatter(
        x = [0.52],
        y = [0.10],
        mode="text",
        text=["Strong Similarity Zone"],
        textposition="top right",
        showlegend=False,
        ))

    # Model level data
    #print(time() - start, '6 callback')
    return [
            disable_protein_filter,
            disable_cluster_filter,
			disable_count_filter,
	        scatter_fig, 
	        #table_dff.to_dict('records'),
			num_hist,
			length_hist,
            cluster_view, 
            confidence_view,
			functional_view,
			cluster_label,
			cluster_click_data,
           ]

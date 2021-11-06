from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
import app1
#import app1_baseline


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.H1("Welcome to AlphaFold"),
    dcc.Link('Open Structural Similarity Explorer', href='/app1'),
    #dcc.Link('Open Structural Similarity Explorer - Baseline', href='/app1_baseline'),
    html.Br(),
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/app1':
        return app1.layout
    #elif pathname == '/apps/app2':
    #    return app2.layout
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=False)

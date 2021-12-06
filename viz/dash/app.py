import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__,
    suppress_callback_exceptions=True,
    #external_stylesheets=[dbc.themes.SOLAR],
    external_stylesheets=[dbc.themes.QUARTZ],
    )
# Add server
server = app.server

import dash
from config import Config

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [Config.style]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, use_pages=True)

server = app.server
app.config.suppress_callback_exceptions = True
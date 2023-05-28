import dash_bootstrap_components as dbc
from dash import dcc, dash
from dash import html
from dash.dependencies import Input, Output, State

from app import app
from ops.opapp import Util

read_process_status = ""

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/home"),
        dbc.DropdownMenuItem("Expense", href="/expense"),
        dbc.DropdownMenuItem("Income", href="/income"),
        dbc.DropdownMenuItem("Balance", href="/balance"),
        dbc.DropdownMenuItem("Debt", href="/debt"),
        dbc.DropdownMenuItem("InvestiSight", href="/investisight"),
        dbc.DropdownMenuItem("Net Worth", href="/networth"),
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        # dbc.Col(html.Img(src="/assets/virus.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("eLiqSys", className="ml-2")),
                    ],
                    align="center",
                    # no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    # home.layout,
    dash.page_container,
])

util = Util()
read_process_status = util.read_data()
from dash import html
import dash_bootstrap_components as dbc
from config import Config
import dash

dash.register_page(__name__, path='/home')


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Welcome to eLiqSys", className="text-center")
                    , className="mb-5 mt-5")
        ]),

        dbc.Row([
            dbc.Col(html.H5(children='The four states of Liquid')
                    , className="mb-5 text-center")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[
                                        html.H4(Config.expense_title, className="card-title text-center"),
                                        html.P(children=Config.expense_desc, className="text-center"),
                                        html.A(href="/expense", className="stretched-link"),
                                       ],
                            body=True, color="dark", outline=True, style={"height": Config.card_height}),
            className="mb-4"),

            dbc.Col(dbc.Card(children=[
                html.H4(Config.income_title, className="card-title text-center"),
                html.P(children=Config.income_desc, className="text-center"),
                html.A(href="/income", className="stretched-link"),
            ],
                body=True, color="dark", outline=True, style={"height": Config.card_height}),
                className="mb-4"),

            # dbc.Col(dbc.Card(children=[
            #                             html.H4(Config.balance_title, className="card-title text-center"),
            #                             html.P(children=Config.balance_desc, className="text-center"),
            #                             html.A(href="/balance", className="stretched-link"),
            #                            ],
            #                 body=True, color="dark", outline=True, style={"height": Config.card_height}),
            # className="mb-4"),

            dbc.Col(dbc.Card(children=[
                                        html.H4(Config.debt_title, className="card-title text-center"),
                                        html.P(children=Config.debt_desc, className="text-center"),
                                        html.A(href="/debt", className="stretched-link"),
                                       ],
                            body=True, color="dark", outline=True, style={"height": Config.card_height}),
            className="mb-4"),
        ], className="mb-5", justify="center"),

        dbc.Row([
                    dbc.Col(dbc.Card(children=[
                                                html.H4(Config.invest_title, className="card-title text-center"),
                                                html.P(children=Config.invest_desc, className="text-center"),
                                                html.A(href="/investisight", className="stretched-link"),
                                               ],
                                    body=True, color="dark", outline=True, style={"width": 500,"height": Config.card_height}),
                    className="mb-4", width='auto'),
                    dbc.Col(dbc.Card(children=[
                                                html.H4(Config.budget_title, className="card-title text-center"),
                                                html.P(children=[Config.budget_desc], className="text-center"),
                                                html.A(href="/budget", className="stretched-link"),
                                               ],
                                    body=True, color="dark", outline=True, style={"width": 500,"height": Config.card_height}),
                    className="mb-4", width='auto'),
                    dbc.Col(dbc.Card(children=[
                                                html.H4(Config.net_worth_title, className="card-title text-center"),
                                                html.A(href="/networth", className="stretched-link"),
                                               ],
                                    body=True, color="dark", outline=True, style={"width": 500,"height": str(Config.card_height - 50)}),
                    className="mb-4", width='auto'),
            ], justify="center"),

    ])

])



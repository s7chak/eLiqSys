import os

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc
from dash import html

from config import Config
from ops.opapp import Util, Cleaner

dash.register_page(__name__, path='/balance')

print("Balance page opened.")
file = Config.wd_path+Config.data_csv
if not os.path.isfile(file):
    util = Util()
    data_status = util.read_data()
    if 'Success' not in data_status:
        print(data_status)
        print(Config.line_string)
df = pd.read_csv(file)
period = df[Config.date_col].min()+" - " + df[Config.date_col].max()

df[Config.date_col] = pd.to_datetime(df[Config.date_col])
cleaner = Cleaner()
savings = cleaner.prepare_dataframe(df, Config.savings_fields, Config.date_col, Config.balance)
savings[Config.balance] = savings[Config.balance].astype('str')
savings[Config.balance] = savings[Config.balance].str[:-1].astype('float64')
balance_df = savings[savings[Config.balance].notna()]

sign = 1
key = Config.balance
report_data = {key:{}}
report_data[key][Config.appendix_title]={}
plots={}
plot_num = 0
image_path = Config.today_folder+Config.plot_path_prefix

print("File read for "+key+" analysis")
print(Config.line_string)
if not os.path.exists(Config.today_folder):
    os.mkdir((Config.today_folder))
if not os.path.exists(Config.today_folder+Config.plot_path_prefix):
    os.mkdir(Config.today_folder+Config.plot_path_prefix)
    print("Reporting folder created for today:"+Config.today_folder)


util = Util()



def do_balance_analysis(d, key):
    # tables
    plot_num=0
    mean_grp_balance = util.get_average_multi(d, [Config.ym_col, Config.acc_col], Config.balance)
    eom_grp_balance = round(d.groupby([Config.ym_col, Config.acc_col])[Config.ym_col, Config.acc_col, Config.balance].tail(1))
    month_tots = round(pd.DataFrame(eom_grp_balance.groupby(Config.ym_col)[Config.balance].sum()))
    mean_month_tots = round(pd.DataFrame(mean_grp_balance.reset_index().groupby(Config.ym_col)[[Config.balance]].sum()))
    report_data[key][Config.appendix_title] = {}
    report_data[key][Config.appendix_title]["Total EOM Balances for whole period: \n"] = month_tots.reset_index()


    acc_wise = pd.DataFrame(mean_grp_balance).reset_index()
    # fig1.add_trace(go.Scatter(x=acc_wise[Config.ym_col].values, y=acc_wise[Config.balance].values))
    fig1 = px.line(acc_wise,x=Config.ym_col, y=Config.balance, color=Config.acc_col)
    plot_num += 1
    plots[plot_num] = {}
    plots[plot_num][Config.plot_word] = fig1

    summary = []
    summary.append("Current Total Liquids: " + "${:,.2f}".format(month_tots.tail(1).values[0][0])+"\n")
    summary.append("Mean EOM Balance: "+"${:,.2f}".format(mean_month_tots.mean().values[0])+"\n")


    return summary



balance_summary = do_balance_analysis(balance_df, key)







def serve_layout():
    layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H1(children=key+' Analysis'), className="mb-2"),
                dbc.Col(html.H4(children='Period: '+period), className="mb-2", width='auto')
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=[html.H6("Summary")] + balance_summary
                             , body=True, color="dark", outline=True, style={"width": 800}, className="text-center"),
                ),
                dbc.Col(dbc.Card(children=
                         [dcc.Graph(figure=plots[i][Config.plot_word], id='plot_' + str(i)) for i in plots.keys()]
                         , body=True, color="dark", outline=True, style={"width": 800}, className="text-center"),
                        )
            ]),


        ])
    ])

    return layout


layout = serve_layout()


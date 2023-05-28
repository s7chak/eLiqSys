import copy
import os

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from dash import html

from config import Config
from ops.opapp import Util, Cleaner
import dash

dash.register_page(__name__, path='/debt')
print("Debt page opened.")
df = pd.read_csv(Config.wd_path+Config.final_data_file, sep=',')
period = df[Config.date_col].min()+" - " + df[Config.date_col].max()


df[Config.date_col] = pd.to_datetime(df[Config.date_col])
cleaner = Cleaner()
tf = cleaner.prepare_dataframe(df, Config.transaction_use_fields, Config.date_col, Config.amount)
dbt = copy.deepcopy(tf[(tf[Config.category] == Config.debtrepay) | (tf[Config.subcategory].isin(["AutoLoan"]))])
dbt = dbt[~dbt[Config.subcategory].isin(['CCPayment'])]
inc = pd.DataFrame(tf[tf[Config.category].isin(Config.incomes)])

sign = -1
key = Config.debt_title
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



def do_debt_analysis(d, inc, key):
    # tables
    plot_num=0
    tot_debt_monthly = sign * util.get_sum(d, Config.ym_col, Config.amount)
    report_data[key][Config.appendix_title] = {}
    report_data[key][Config.appendix_title][
        "Grouped " + key + " per " + Config.ym_col + " the Totals for whole period: \n"] = tot_debt_monthly.reset_index()
    tot_subcategory = util.get_sum_multi(d, [Config.ym_col, Config.subcategory], Config.amount).reset_index()
    tot_subcategory[Config.amount] = sign * tot_subcategory[Config.amount]
    report_data[key][Config.appendix_title]["Grouped " + key + " per " + Config.subcategory + " per month: \n"] = tot_subcategory[[Config.ym_col, Config.subcategory, Config.amount]].reset_index()
    x = tot_subcategory.pivot(index=Config.ym_col, columns=Config.subcategory, values=Config.amount).fillna(0)
    fig1 = go.Figure(
        data=
            [go.Bar(name=s, x=x.index, y=x[s],offsetgroup=i+1) for i, s in enumerate(x.columns)]
        ,
    )
    plot_num += 1
    plots[plot_num] = {}
    plots[plot_num][Config.plot_word] = fig1

    gi = inc.groupby([Config.ym_col])[Config.amount].sum().reset_index()
    d = - pd.DataFrame(d.groupby([Config.ym_col])[Config.amount].sum())
    fig2 = go.Figure(
        data=[
            go.Scatter(x=gi[Config.ym_col].values, y=gi[Config.amount].values, name="Income"),
            go.Scatter(x=d.index, y=d[Config.amount].values, name="Debt Payments")
        ],
    )
    plot_num += 1
    plots[plot_num] = {}
    plots[plot_num][Config.plot_word] = fig2


do_debt_analysis(dbt, inc, key)







def serve_layout():
    plot_style = {'width': 1000,
                      "margin-left": "auto",
                      "margin-right": "auto",
                      }

    layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H1(children=key+' Analysis'), className="mb-2"),
                dbc.Col(html.H4(children='Period: '+period), className="mb-2", width='auto')
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=
                        [dcc.Graph(figure=plots[i][Config.plot_word], id='plot_'+str(i)) for i in plots.keys()]
                        , body=True, color="dark", outline=True, style={"width": 1000}, className="text-center"),
                )], justify='center')
        ])
    ])

    return layout


layout = serve_layout()


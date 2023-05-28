import os
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from dash import html

from config import Config
from ops.opapp import Util

dash.register_page(__name__, path='/income')
print("Income page opened.")
df = pd.read_csv(Config.wd_path+Config.final_data_file, sep=',')

sign = 1
key = Config.income_col
period = df[Config.date_col].min()+" - " + df[Config.date_col].max()
report_data = {key:{}}
report_data[key][Config.appendix_title]={}
plots={}
txt_data={}
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



def do_inc_exp_plot_analysis(d, key):
    # tables
    plot_num=0
    inc_exp=pd.DataFrame(d[d[Config.category].isin(Config.incomes)].groupby([Config.ym_col])[Config.amount].sum())
    inc_exp.rename(columns={Config.amount:Config.income_col}, inplace=True)
    inc_exp[Config.expense_col]=pd.DataFrame(-d[d[Config.category].isin(Config.expenses)].groupby([Config.ym_col])[Config.amount].sum())
    inc_exp[Config.gain_col]=inc_exp[Config.income_col] - inc_exp[Config.expense_col]
    pos = 'positive'
    conditions = [(inc_exp[Config.income_col] > inc_exp[Config.expense_col]),
                  (inc_exp[Config.income_col] < inc_exp[Config.expense_col]),
                  (inc_exp[Config.income_col] == inc_exp[Config.expense_col])]
    choices = ['Positive', 'Negative', 'Breakeven']
    inc_exp[pos] = np.select(conditions, choices, default='Positive')
    # plots
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.income_col].values, name=Config.income_col))
    fig2.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.expense_col].values, name=Config.expense_col))
    fig2.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.gain_col].values, name=Config.gain_col))
    plot2 = image_path + '/' + key + '2.png'
    plot_num += 1
    plots[plot_num] = {}
    plots[plot_num][Config.plot_word] = fig2
    report_data[key]["Income, Expense, Gain Monthly plot"] = fig2
    report_data[key]["Gains : Income vs Expense \n"] = inc_exp

    fig1 = go.Figure()
    poss = inc_exp[inc_exp[pos] == 'Positive']
    negs = inc_exp[inc_exp[pos]=='Negative']
    fig1.add_trace(go.Scatter(x=poss[Config.income_col].values, y=poss[Config.expense_col].values, mode='markers', name='Positive Months', text=poss.index))
    fig1.add_trace(go.Scatter(x=negs[Config.income_col].values, y=negs[Config.expense_col].values, mode='markers',name='Negative Months', text=negs.index))
    fig1.update_layout(legend_title_text="Income vs Expense Months")
    m = max(inc_exp[Config.income_col].max(), inc_exp[Config.expense_col].max())
    fig1.add_trace(go.Scatter(x=np.arange(0,m), y=np.arange(0,m), name=Config.income_col+' = '+Config.expense_col))
    m_exp = inc_exp[inc_exp[Config.income_col] < inc_exp[Config.expense_col]]
    m_inc = inc_exp[inc_exp[Config.income_col] > inc_exp[Config.expense_col]]
    report_data[key][Config.appendix_title]={}
    d1 = "Greater Expense than Income months"
    d2 = "Greater Income than Expense months"
    report_data[key][Config.appendix_title][d1] = m_exp.reset_index()
    report_data[key][Config.appendix_title][d2] = m_inc.reset_index()
    txt_data[d1] = ", ".join(m_exp.index.values)
    txt_data[d2] = ", ".join(m_inc.index.values)
    plot1 = image_path+'/'+key+'1.png'
    plot_num += 1
    plots[plot_num] = {}
    plots[plot_num][Config.plot_word] = fig1
    report_data[key]["Income vs Expense Months Scatter plot"] = plot1

    cyear = str(datetime.today().year)
    cmonth = str(datetime.today().month)
    d3 = "Total Gain in whole period:   "
    report_data[key][d3] = inc_exp[Config.gain_col].sum()
    txt_data[d3] = str("${:,.2f}".format(inc_exp[Config.gain_col].sum()))
    d4 = "YTD Gain:   "
    d5 = "This month gain :   "
    inc_exp.reset_index(inplace=True)
    inc_exp[Config.year_col] = inc_exp[Config.ym_col].str.slice(0,4)
    d4data = "${:,.2f}".format(inc_exp[inc_exp[Config.year_col]==cyear][Config.gain_col].sum())
    d5data = "${:,.2f}".format(inc_exp[inc_exp[Config.ym_col]==cyear+'-'+cmonth.rjust(2, '0')][Config.gain_col].sum())
    txt_data[d4] = str(d4data)
    txt_data[d5] = str(d5data)
    # report_data[key]["Gain in current year: $info"] = inc_exp[inc_exp[Config.ym_col==cyear+'-'+cmonth.rjust(2, '0')]][Config.gain_col].sum()
    # report_data[key]["Gain in current month: $info"] = inc_exp[inc_exp[Config.ym_col==cyear]][Config.gain_col].sum()


do_inc_exp_plot_analysis(df, key)







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
                                 [html.P(i+txt_data[i], id='txt_title_' + str(i)) for i in txt_data.keys()]
                                 , body=True, color="dark", outline=True, style={"width": 400}, className="text-center"),
                    ),
                    dbc.Col(
                        dbc.Card(children=
                                 [dcc.Graph(figure=plots[i][Config.plot_word], id='plot_'+str(i)) for i in plots.keys()]
                                 , body=True, color="dark", outline=True, style={"width": 800}, className="text-center"),
                    ),
                ])
        ])
    ])

    return layout


layout = serve_layout()


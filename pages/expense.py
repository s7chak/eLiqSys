import os
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from dash import dcc, callback
from dash import html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config import Config
from ops.opapp import Util, Cleaner

dash.register_page(__name__, path='/expense')
print("Expense page opened.")

df = pd.read_csv(Config.wd_path+Config.final_data_file, sep=',')
tf = df[df[Config.category].isin(Config.expenses)]
period = df[Config.date_col].min()+" - " + df[Config.date_col].max()
cleaner = Cleaner()
util = Util()
tf = cleaner.prepare_dataframe(tf, Config.transaction_use_fields, Config.date_col, Config.amount)
sign = -1
key = Config.expense_col
report_data = {key:{}}
report_data[key][Config.appendix_title]={}
top_1_plots={}
top_plots={}
mid_plots={}
catplots={}
plot_num=0
txt_data={}
image_path = Config.today_folder+Config.plot_path_prefix

print("File read for "+key+" analysis")
print(Config.line_string)
if not os.path.exists(Config.today_folder):
    os.mkdir((Config.today_folder))
if not os.path.exists(Config.today_folder+Config.plot_path_prefix):
    os.mkdir(Config.today_folder+Config.plot_path_prefix)
    print("Reporting folder created for today:"+Config.today_folder)


def ax_plot(trace, xlabel, ylabel, title):
    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_layout(legend_title_text=title)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    return fig

def add_plot(pts, fig):
    plot_num = len(pts.keys())
    pts[plot_num]={}
    pts[plot_num][Config.plot_word] = fig

def get_cat_subcats(d):
    cat_title = ", ".join(list(d[Config.category].unique()))
    subcat_title = ", ".join(list(d[Config.subcategory].unique()))
    return cat_title, subcat_title

def do_regression(d, x, y, key, ind):
    models = [sm.OLS, LinearRegression, DecisionTreeRegressor, XGBRegressor]
    lines = []
    for model in models:
        if model.__name__ in ['OLS']:
            fit_results = model(y.values, sm.add_constant(x.values)).fit().predict()
        else:
            fit_results = model().fit(sm.add_constant(x.values), y.values).predict(sm.add_constant(x.values))
        lines.append(go.Scatter(x=ind, y=fit_results, name=model.__name__+"-"+key))
    return lines


def do_inc_trend_analysis(d, sign, key, suf):
    inc_exp = pd.DataFrame(d[d[Config.category].isin(Config.incomes)].groupby([Config.ym_col])[Config.amount].sum())
    indispensable = Config.regular_expenses
    indis = d[d[Config.category].isin(indispensable)]
    indis = sign * util.get_sum(indis, Config.ym_col, Config.amount)
    inc_exp[Config.indispensable_expense] = indis[Config.amount]
    exp = d[d[Config.category].isin(Config.expenses)]
    exp = sign * util.get_sum(exp, Config.ym_col, Config.amount)
    inc_exp[Config.expense_col] = exp[Config.amount]
    inc_exp[Config.gain_col] = inc_exp[Config.amount] - inc_exp[Config.expense_col]
    inc_exp[Config.gain_col+'_indispensable'] = inc_exp[Config.amount] - inc_exp[Config.expense_col]
    inc_exp.index = pd.to_datetime(inc_exp.index)
    gain_lines = do_regression(inc_exp, inc_exp[Config.amount], inc_exp[Config.gain_col], Config.gain_col, inc_exp.index)
    exp_lines = do_regression(inc_exp, inc_exp[Config.amount], inc_exp[Config.expense_col], Config.expense_col, inc_exp.index)
    gain_lines.append(go.Bar(x=inc_exp.index, y=inc_exp[Config.amount].values, name=Config.income_col))
    fig1 = go.Figure()
    for l in gain_lines+exp_lines:
        fig1.add_trace(l)
    yplot1 = image_path + key + suf +'1.png'
    fig1.write_image(yplot1)
    add_plot(top_1_plots, fig1)
    print("Gain, Expense trends done.")



def do_year_analysis(d, sign, key, suf):
    # sum of expenses per year
    tot_category = sign*util.get_sum(d, Config.year_col, Config.amount)
    fig1 = ax_plot(go.Bar(x=tot_category.index, y=tot_category[Config.amount].values,), Config.year_col, Config.amount, "Total Expense per Year")
    fig1.update_layout({'width':500})
    yplot1 = image_path + key + suf +'1.png'
    fig1.write_image(yplot1)
    add_plot(top_plots, fig1)
    print('Yearly plot done.')

    #categorical
    num_years = len(d[Config.year_col].unique())
    fig2 = make_subplots(rows=num_years, cols=1, vertical_spacing=0.03)
    d[Config.year_col] = d[Config.year_col].astype(str)
    for i in range(num_years):
        y = d[Config.year_col].unique()[i]
        s = sign*d[d[Config.year_col]==y].groupby(Config.category)[Config.amount].sum().sort_values()
        fig2.add_trace(go.Bar(x=s.index,y=s.values,name=y), row=i+1, col=1)
        fig2.update_layout(height=900, title=go.layout.Title(text="Year-wise expense category totals"))
        report_data[key][Config.appendix_title]["Categorical "+key+" per "+Config.year_col +": \n"] = s.T
    yplot2 = image_path+key+suf +'2.png'
    fig2.write_image(yplot2)
    add_plot(mid_plots, fig2)
    cyear = datetime.today().year
    ytd = "${:,.2f}".format(tot_category[tot_category.index==cyear].values[0][0])
    txt_data["YTD Expense: "] = ytd
    print('Yearly Categorical plot done.')
    # report_data[key]["Total "+name+" plots per "+Config.category+" plot"] = yplot2


def do_category_analysis(d, sign, key, suf):
    # sum of expenses per year
    tot_category = sign*util.get_sum(d, Config.category, Config.amount).sort_values(Config.amount)
    tot_subcategory = sign*util.get_sum(d, Config.subcategory, Config.amount).sort_values(Config.amount)
    mean_monthly_category = sign * util.get_average_multi(d, [Config.ym_col, Config.category], Config.amount)
    mean_monthly_subcategory = sign * util.get_average_multi(d, [Config.ym_col, Config.subcategory], Config.amount)
    fig1 = ax_plot(go.Bar(x=tot_category.index, y=tot_category[Config.amount].values, ), Config.category, Config.amount, "Total Expense per Category for full period")
    fig2 = ax_plot(go.Bar(x=tot_subcategory.index, y=tot_subcategory[Config.amount].values, ), Config.subcategory, Config.amount, "Total Expense per Subcategory for full period")
    yplot1 = image_path + key+Config.category + suf +'1.png'
    yplot2 = image_path + key + Config.subcategory + suf +'1.png'
    fig1.write_image(yplot1)
    fig2.write_image(yplot2)
    add_plot(catplots, fig1)
    add_plot(subcatplots, fig2)
    print('Total categorical plot for whole period done.')

    #pie categories
    fig2 = ax_plot(go.Pie(labels=tot_category.index, values=tot_category[Config.amount]),'','','Categorical Pie')
    yplot2 = image_path + key + suf + '-pie-1.png'
    fig2.write_image(yplot2)
    add_plot(top_1_plots, fig2)


    #categorical
    num_categories = len(d[Config.category].unique())
    fig3 = make_subplots(rows=num_categories, cols=1, vertical_spacing=0.03, subplot_titles=list(d[Config.category].unique()))
    for i,c in enumerate(d[Config.category].unique()):
        s = sign*d[d[Config.category]==c].groupby(Config.ym_col)[Config.amount].sum().sort_values()
        fig3.add_trace(
                        go.Bar(x=s.index,y=s.values,name=c), row=i+1,col=1,
                    )
        fig3.add_trace(
            go.Scatter(x=s.index, y=[s.values.mean() for x in s.index], name=c+' mean'), row=i + 1, col=1,
        )
        # report_data[key][Config.appendix_title]["Categorical "+key+" per "+Config.ym_col +": \n"] = s.T
    fig3.update_layout(height=250*num_categories)
    yplot3 = image_path+key+suf +'3.png'
    fig3.write_image(yplot3)
    add_plot(catplots, fig3)
    print('Monthly Categorical plot done.')
    # report_data[key]["Total "+name+" plots per "+Config.category+" plot"] = yplot2


    # Subcategorical
    num_subcategories = len(d[Config.subcategory].unique())
    fig4 = make_subplots(rows=num_subcategories, cols=1, vertical_spacing=0.03,
                         subplot_titles=list(d[Config.subcategory].unique()))
    for i, c in enumerate(d[Config.subcategory].unique()):
        s = sign * d[d[Config.subcategory] == c].groupby(Config.ym_col)[Config.amount].sum().sort_values()
        fig4.add_trace(
            go.Bar(x=s.index, y=s.values, name=c), row=i + 1, col=1,
        )
        fig4.add_trace(
            go.Scatter(x=s.index, y=[s.values.mean() for x in s.index], name=c+' mean'), row=i + 1, col=1,
        )
        # report_data[key][Config.appendix_title]["Categorical "+key+" per "+Config.ym_col +": \n"] = s.T
    fig4.update_layout(height=250 * num_subcategories)
    yplot4 = image_path + key + suf +'4.png'
    fig4.write_image(yplot4)
    add_plot(subcatplots, fig4)
    print('Monthly Categorical plot done.')
    # report_data[key]["Total "+name+" plots per "+Config.category+" plot"] = yplot2


def do_exp_analysis(d, sign, key):
    inc_exp = pd.DataFrame(d[d[Config.category].isin(Config.incomes)].groupby([Config.ym_col])[Config.amount].sum())
    indispensable = Config.regular_expenses
    indis = d[d[Config.category].isin(indispensable)]
    indis = sign*util.get_sum(indis, Config.ym_col, Config.amount)
    inc_exp[Config.indispensable_expense] = indis[Config.amount]
    exp = d[d[Config.category].isin(Config.expenses)]
    exp = sign * util.get_sum(exp, Config.ym_col, Config.amount)
    inc_exp[Config.expense_col] = exp[Config.amount]
    txt_data["Indispensable Expense Categories:  "] = ",  ".join(indispensable)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.amount].values, name=Config.income_col))
    fig1.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.indispensable_expense].values, name=Config.indispensable_expense))
    fig1.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.expense_col].values, name=Config.expense_col))
    yplot1 = image_path + key + '1.png'
    fig1.write_image(yplot1)
    add_plot(top_plots, fig1)

    inc_exp[Config.gain_col] = inc_exp[Config.amount] - inc_exp[Config.indispensable_expense]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=inc_exp.index, y=inc_exp[Config.gain_col].values, name=Config.gain_col))
    fig1.add_trace(go.Scatter(x=inc_exp.index, y=[inc_exp[Config.indispensable_expense].mean() for i in inc_exp.index], name=Config.indispensable_expense+' Monthly Mean'))
    yplot1 = image_path + key + '2.png'
    fig1.write_image(yplot1)
    add_plot(top_plots, fig1)

    print("Income vs indispensable Income done.")

do_inc_trend_analysis(df, sign, key,'-trend')
do_year_analysis(tf, sign, key,'-yearly')
do_exp_analysis(df, sign, key)
subcatplots = {}
do_category_analysis(tf, sign, key,'-cat')



cat_list = Config.regular_expenses
cat_dropdown = dcc.Dropdown(
                options=[i for i in tf[Config.category].unique()],
                value= cat_list,
                multi=True,
                id='cat-dropdown-id',
                placeholder="Select categories",
                style={'width': '100%', 'color':'black'}
            ),

subcat_list = []
subcat_dropdown = dcc.Dropdown(
                options=[i for i in tf[Config.subcategory].unique()],
                value= subcat_list,
                multi=True,
                id='subcat-dropdown-id',
                placeholder="Select categories",
                style={'width': '100%', 'color':'black'}
            ),




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
                             [dcc.Graph(figure=top_1_plots[i][Config.plot_word], id='plot_top_1_' + str(i)) for i in top_1_plots.keys()]
                             , body=True, color="dark", outline=True, style={"width": 1300, "height": 1000},
                             className="text-center"),
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=
                             [html.P(i + txt_data[i], id='txt_title_' + str(i)) for i in txt_data.keys()] +
                             [dcc.Graph(figure=mid_plots[i][Config.plot_word], id='plot_mid_' + str(i)) for i in mid_plots.keys()]
                             , body=True, color="dark", outline=True, style={"width": 620, "height":1400}, className="text-center"),
                ),
                dbc.Col(
                    dbc.Card(children=
                             [dcc.Graph(figure=top_plots[i][Config.plot_word], id='plot_top_' + str(i)) for i in top_plots.keys()]
                             , body=True, color="dark", outline=True, style={"width": 620, "height":1400}, className="text-center"),align='center'
                ),
            ],),

            dbc.Row([
                dbc.Col(
                    cat_dropdown
                , style={"width": 600}, className="text-center"),
                dbc.Col(
                    subcat_dropdown
                    , style={"width": 600}, className="text-center"),
            ]),
            dbc.Row([
                dbc.Card(children=[
                    html.H5("Categories Monthly")
                ] +
                [
                    dcc.Graph(id='cat_plot_1'),
                ])
            ]),
            dbc.Row([
                dbc.Card(children=[
                                      html.H5("Subcategories Monthly")
                                  ] +
                                  [
                                      dcc.Graph(id='subcat_plot_1'),
                                  ])
            ]),


            dbc.Row([
                dbc.Col(
                    dbc.Card(children=
                             [dcc.Graph(figure=catplots[i][Config.plot_word], id='plot_' + str(i)) for i in
                              catplots.keys()]
                             , body=True, color="dark", outline=True, style={"width": 600}, className="text-center"), ),
                dbc.Col(
                    dbc.Card(children=
                             [dcc.Graph(figure=subcatplots[i][Config.plot_word], id='plot_' + str(i)) for i in
                              subcatplots.keys()], body=True, color="dark", outline=True, style={"width": 600},
                             className="text-center"), )
            ]),


            dbc.Row([
                dbc.Col(dbc.Card(children=[
                    html.H4("Categories", className="card-title text-center"),
                    html.P(children=get_cat_subcats(df)[0], className="text-center")
                ],
                    body=True, color="dark", outline=True, style={"height": Config.card_height}),
                    className="mb-4"),

                dbc.Col(dbc.Card(children=[
                    html.H4("Sub-categories", className="card-title text-center"),
                    html.P(children=get_cat_subcats(df)[1], className="text-center"),
                ],
                    body=True, color="dark", outline=True, style={"height": Config.card_height}),
                    className="mb-4"),

            ])
        ])
    ])


    return layout


layout = serve_layout()




@callback(
    Output(component_id='cat_plot_1',component_property='figure'),
    [Input(component_id='cat-dropdown-id', component_property='value')]
)
def category_analysis(value):
    if len(value) > 0:
        sign=-1
        catdf = tf[tf[Config.category].isin(value)]
        cat_traces = []
        fig = go.Figure()
        for c in value:
            y = sign*catdf[catdf[Config.category]==c].groupby(Config.ym_col).sum()
            cat_traces.append(go.Bar(x=y.index, y=y[Config.amount].values,name=c))
        monthly = sign*catdf.groupby(Config.ym_col)[Config.amount].sum()
        mean = monthly.mean()
        cat_traces.append(go.Scatter(x=monthly.index, y=[mean for i in monthly.index], name='Monthly Mean'))
        fig.add_traces(cat_traces)
        return fig
    else:
        return go.Figure()



@callback(
    Output('subcat-dropdown-id','options'),
    Output('subcat-dropdown-id','value'),
    [Input(component_id='cat-dropdown-id', component_property='value')]
)
def select_subcategories(catvalue):
    if len(catvalue) > 0:
        catdf = tf[tf[Config.category].isin(catvalue)]
        subcats = list(catdf[Config.subcategory].unique())
        return subcats, subcats
    else:
        return [], []


@callback(
    Output(component_id='subcat_plot_1',component_property='figure'),
    [Input(component_id='cat-dropdown-id', component_property='value'),
     Input(component_id='subcat-dropdown-id', component_property='value')]
)
def subcategory_analysis(catvalue, subcatvalue):
    if len(catvalue) > 0:
        sign=-1
        catdf = tf[tf[Config.category].isin(catvalue)]
        if len(subcatvalue) > 0:
            catdf = catdf[catdf[Config.subcategory].isin(subcatvalue)]
        cat_traces = []
        fig = go.Figure()
        for c in catvalue:
            scvalues = catdf[Config.subcategory].unique()
            for sc in scvalues:
                y = sign*catdf[(catdf[Config.category]==c) & (catdf[Config.subcategory]==sc)].groupby(Config.ym_col).sum()
                cat_traces.append(go.Bar(x=y.index, y=y[Config.amount].values,name=c+':'+sc))
        monthly = sign*catdf.groupby(Config.ym_col).sum()
        mean = monthly[Config.amount].mean()
        cat_traces.append(go.Scatter(x=monthly.index, y=[mean for i in monthly.index], name='Monthly Mean'))
        fig.add_traces(cat_traces)
        return fig
    else:
        return go.Figure()



print(Config.line_string)

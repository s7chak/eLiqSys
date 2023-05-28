import sys

import dash_bootstrap_components as dbc
from dash import dcc, callback, Input, Output
from dash import html
import dash
import pandas as pd
import plotly.graph_objects as go
from config import Config
from ops.opapp import Util, Cleaner
import datetime as dt
dash.register_page(__name__, path='/budget')


df = pd.read_csv(Config.wd_path+Config.final_data_file, sep=',')
start = df[Config.date_col].min()
end = df[Config.date_col].max()
period = start+" - " + end
cleaner = Cleaner()
util = Util()
sign = -1
key = Config.budget_title
tf = df[df[Config.category].isin(Config.expenses)]
tf = cleaner.prepare_dataframe(tf, Config.transaction_use_fields, Config.date_col, Config.amount)
report_data = {key:{}}
report_data[key][Config.appendix_title]={}
top_plots={}
txt_data={}
image_path = Config.today_folder+Config.plot_path_prefix
total_added = 0
period_filter = []

def add_plot(pts, fig):
    plot_num = len(pts.keys())
    pts[plot_num]={}
    pts[plot_num][Config.plot_word] = fig


cat_list = Config.regular_expenses
cat_dropdown = dcc.Dropdown(
                options=[i for i in tf[Config.category].unique()],
                value= cat_list,
                multi=True,
                id='bcat-dropdown-id',
                placeholder="Select categories",
                style={'width': '100%', 'color':'black'}
            ),

subcat_list = []
subcat_dropdown = dcc.Dropdown(
                options=[i for i in tf[Config.subcategory].unique()],
                value= subcat_list,
                multi=True,
                id='bsubcat-dropdown-id',
                placeholder="Select categories",
                style={'width': '100%', 'color':'black'}
            ),



def serve_layout():
    layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H2(children='Budget'), className="mb-2")
            ]),
            dbc.Row([
                dcc.DatePickerRange(
                    id='period-picker-range',
                    min_date_allowed=dt.date(2019, 1, 1),
                    max_date_allowed=dt.date(2030, 12, 31),
                    initial_visible_month=dt.date.today(),
                    start_date=dt.datetime.strptime(start, '%Y-%m-%d').date(),
                    end_date=dt.date.today()
                ),
                ]),
            dbc.Row([
                dbc.Col(
                    cat_dropdown
                    , style={"width": 600}, className="text-center"),
                dbc.Col(
                    subcat_dropdown
                    , style={"width": 600}, className="text-center"),
            ]),
            dbc.Row([
                    dcc.Input(id="add-monthly-expense", value=0, type="text", placeholder="Enter monthly expense",
                              style={'marginRight': '20px', 'width':'auto'}),
                    dcc.Input(id="add-6monthly-expense", value=0, type="text", placeholder="Enter 6-monthly expense",
                              style={'marginRight': '20px', 'width':'auto'}),
                    dcc.Input(id="add-yearly-expense", value=0, type="text", placeholder="Enter yearly expense",
                              style={'marginRight': '20px', 'width':'auto'}),
            ], justify='center'),
            dbc.Row([
                html.P(id="total_added_id")
                ]),
            dbc.Row([
                dbc.Card(children=[
                                      html.H5("Categories Monthly")
                                  ] +
                                  [
                                      dcc.Graph(id='budgcat_plot_1'),
                                  ])
            ]),
            dbc.Row([
                dbc.Card(children=[
                                      html.H5("Subcategories Monthly")
                                  ] +
                                  [
                                      dcc.Graph(id='budgsubcat_plot_1'),
                                  ])
            ]),
            dbc.Row([
                dbc.Col(html.H6(children='Checking mean expense forecast and future income.'), className="mb-4")
            ]),
        ])
    ])

    return layout

layout = serve_layout()


def get_year_added_expense(sign,m,m6,y):
    added = 0
    try:
        m = sign*int(m)
        m6 = sign*int(m6)
        y = sign*int(y)
    except:
        print("Non integer typed. "+sys.exc_info())
    if m != 0:
        added += m * 12
    if m6 != 0:
        added += m6 * 2
    if y != 0:
        added += y
    return added

def add_expense(df, ex):
    years = list(df.Year.unique())
    desc = "NewExpense"
    ex = float(ex)
    for y in years:
        df.loc[-1] = [dt.datetime.strptime(str(y)+"-01-01", '%Y-%m-%d'), desc, ex, 'H','Chase',desc,desc,8190,y,1,1,'Monday',str(y)+'-01']
        df.index = df.index + 1
        df.sort_values(Config.date_col, inplace=True)

    print(str(ex) + " expense added.")


@callback(
    Output(component_id='budgcat_plot_1',component_property='figure'),
    Output(component_id='total_added_id',component_property='value'),
    [Input(component_id='bcat-dropdown-id', component_property='value'),
     Input(component_id='add-monthly-expense', component_property='value'),
     Input(component_id='add-6monthly-expense', component_property='value'),
     Input(component_id='add-yearly-expense', component_property='value'),
     Input(component_id='period-picker-range', component_property='start_date'),
     Input(component_id='period-picker-range', component_property='end_date')
     ]
)
def category_analysis(value,m,m6,y, s, e):
    if len(value) > 0:
        sign=-1
        catdf = tf[tf[Config.category].isin(value)]
        if s!=start:
            catdf = catdf[(catdf[Config.date_col]>s) & (catdf[Config.date_col]<e)]
        cat_traces = []
        fig = go.Figure()
        for c in value:
            y_vals = sign*catdf[catdf[Config.category]==c].groupby(Config.ym_col).sum()
            cat_traces.append(go.Bar(x=y_vals.index, y=y_vals[Config.amount].values,name=c))
        added = get_year_added_expense(sign,m, m6, y)
        total_added = 0
        if added:
            add_expense(catdf, added)
            total_added = added
        add_message = "Total added amount: " + str(total_added)
        monthly = sign*catdf.groupby(Config.ym_col)[Config.amount].sum()
        mean = monthly.mean()
        cat_traces.append(go.Scatter(x=monthly.index, y=[mean for i in monthly.index], name='Monthly Mean'))
        fig.add_traces(cat_traces)
        return fig, add_message
    else:
        return go.Figure(), 0



@callback(
    Output('bsubcat-dropdown-id','options'),
    Output('bsubcat-dropdown-id','value'),
    [Input(component_id='bcat-dropdown-id', component_property='value')]
)
def select_subcategories(catvalue):
    if len(catvalue) > 0:
        catdf = tf[tf[Config.category].isin(catvalue)]
        subcats = list(catdf[Config.subcategory].unique())
        return subcats, subcats
    else:
        return [], []


@callback(
    Output(component_id='budgsubcat_plot_1',component_property='figure'),
    [Input(component_id='bcat-dropdown-id', component_property='value'),
     Input(component_id='bsubcat-dropdown-id', component_property='value'),
     Input(component_id='add-monthly-expense', component_property='value'),
     Input(component_id='add-6monthly-expense', component_property='value'),
     Input(component_id='add-yearly-expense', component_property='value'),
     Input(component_id='period-picker-range', component_property='start_date'),
     Input(component_id='period-picker-range', component_property='end_date')
     ]
)
def subcategory_analysis(catvalue, subcatvalue,m,m6,y, s,e):
    if len(catvalue) > 0:
        sign=-1
        catdf = tf[tf[Config.category].isin(catvalue)]
        if s!=start:
            catdf = catdf[(catdf[Config.date_col]>s) & (catdf[Config.date_col]<e)]
        if len(subcatvalue) > 0:
            catdf = catdf[catdf[Config.subcategory].isin(subcatvalue)]
        cat_traces = []
        fig = go.Figure()
        for c in catvalue:
            scvalues = catdf[Config.subcategory].unique()
            for sc in scvalues:
                y_vals = sign*catdf[(catdf[Config.category]==c) & (catdf[Config.subcategory]==sc)].groupby(Config.ym_col)[Config.amount].sum()
                cat_traces.append(go.Bar(x=y_vals.index, y=y_vals,name=c+':'+sc))
        added = get_year_added_expense(sign,m,m6,y)
        if added:
            add_expense(catdf, added)
            total_added = added
        monthly = sign*catdf.groupby(Config.ym_col)[Config.amount].sum()
        mean = monthly.mean()
        cat_traces.append(go.Scatter(x=monthly.index, y=[mean for i in monthly.index], name='Monthly Mean'))
        fig.add_traces(cat_traces)
        return fig
    else:
        return go.Figure()





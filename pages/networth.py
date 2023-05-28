import os

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, Output, callback, Input
from dash import html

from config import Config
from ops.opapp import Util, Cleaner, InvestorFunctions

dash.register_page(__name__, path='/networth')

print("Net Worth page opened.")
file = Config.wd_path+Config.data_csv
if not os.path.isfile(file):
    util = Util()
    data_status = util.read_data()
    if 'Success' not in data_status:
        print(data_status)
        print(Config.line_string)
df = pd.read_csv(file)
period = df[Config.date_col].min()+" - " + df[Config.date_col].max()

use_latest_prices = False

cleaner = Cleaner()
savings = cleaner.prepare_dataframe(df, Config.savings_fields, Config.date_col, Config.balance)
savings[Config.balance] = savings[Config.balance].astype('str')
savings[Config.balance] = savings[Config.balance].str[:-1].astype('float64')
balance_df = savings[savings[Config.balance].notna()]

file = Config.wd_path+Config.inv_final_data_file
if not os.path.isfile(file):
    util = Util()
    data_status = util.read_investment_data()
    if 'Success' not in data_status:
        print(Config.line_string)
invdf = pd.read_csv(file, sep=',')
ret = invdf[invdf[Config.acc_type]==Config.retirement]
bro = invdf[invdf[Config.acc_type]==Config.broker]

sign = 1
key = Config.net_worth_title
report_data = {key:{}}
report_data[key][Config.appendix_title]={}
top_plots={}
growth_plots={}
growth_text = []
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

def add_plot(pts, fig):
    plot_num = len(pts.keys())
    pts[plot_num]={}
    pts[plot_num][Config.plot_word] = fig


def do_asset_networth(liq, bro, ret, key, suf, use_latest):
    monthly_liq = round(liq.groupby([Config.ym_col, Config.acc_col])[Config.ym_col, Config.acc_col, Config.balance].tail(1))
    last_liq = monthly_liq.groupby(Config.ym_col)[Config.balance].sum().tail(1).values[0]
    investor = InvestorFunctions(use_latest)
    brokerage, retirement = investor.analyze_investment_portfolio(bro, ret)
    elems, data = investor.sectype_analysis(brokerage, retirement)
    grouped = round(data.groupby(Config.sec_type_col)[Config.current_amount].sum(), 2)
    inv_total = grouped.sum()
    grouped['Liquid'] = last_liq
    total = grouped.sum()
    fig1 = go.Figure()
    fig1.add_trace(go.Pie(labels=grouped.index, values=grouped.values, hole=0.4))
    fig1.update_layout(
        title_text="Asset Split",
        annotations=[dict(text="${:,.2f}".format(total), x=0.5, y=0.5, font_size=15, showarrow=False)])
    yplot1 = image_path + key + suf + '-pie-1.png'
    fig1.write_image(yplot1)
    add_plot(top_plots, fig1)

    networth_summary = []
    networth_summary.append("Net Assets :  "+"${:,.2f}".format(total)+"\n")
    networth_summary.append("Net Liquid :  " + "${:,.2f}".format(last_liq)+" \t::::\t "+ str(round(last_liq*100/total,1))+"% of assets")
    networth_summary.append("Net Invested :  " + "${:,.2f}".format(inv_total)+" \t::::\t "+ str(round(inv_total*100/total,1))+"% of assets")

    return networth_summary, grouped


def do_liability_networth(key, suf):

    loans = Config.loan_config
    components = pd.DataFrame(loans).T
    sign = -1

    networth_summary = []
    total_debt = components.values.sum()
    networth_summary.append("Net Liability :  "+"${:,.2f}".format(total_debt))
    for i in components.index:
        networth_summary.append(i+" :  " + "${:,.2f}".format(components.loc[i].values[0])+" \t::::\t "+ str(round(components.loc[i].values[0]*100/total_debt,1))+"% of debt")

    fig1 = go.Figure()
    fig1.add_trace(go.Pie(labels=list(loans.keys()), values=[sign*x[0] for x in loans.values()], hole=0.4))
    fig1.update_layout(
        title_text="Liability Split",
        annotations=[dict(text="${:,.2f}".format(sum([sign*x[0] for x in loans.values()])), x=0.5, y=0.5, font_size=15, showarrow=False)])
    yplot1 = image_path + key + suf + '-pie-2.png'
    fig1.write_image(yplot1)
    add_plot(top_plots, fig1)


    return networth_summary, components


def do_growth_analysis(liq, bro, ret, key):
    monthly_liquids = round(liq.groupby([Config.ym_col, Config.acc_col])[Config.ym_col, Config.acc_col, Config.balance].tail(1),2)
    monthly_liquids = monthly_liquids.groupby(Config.ym_col)[Config.balance].sum()

    bro = bro[bro[Config.qty].notna()].groupby([Config.ym_col])[Config.amount].sum()
    ret = ret[ret[Config.qty].notna()].groupby([Config.ym_col])[Config.amount].sum()
    all = pd.merge(monthly_liquids, bro, on=Config.ym_col, how='left').fillna(0)
    all.rename(columns={Config.amount: "bro"}, inplace=True)
    all = pd.merge(all, ret, on=Config.ym_col, how='left').fillna(0)
    all.rename(columns={Config.amount: "ret"}, inplace=True)
    all[Config.broker] = all['bro'].cumsum()
    all[Config.retirement] = all['ret'].cumsum()
    all['Total'] = all[Config.balance] + all[Config.broker] + all[Config.retirement]


    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=all.index, y=all[Config.balance].values, name="Liquid"))
    fig1.add_trace(go.Scatter(x=all.index, y=all[Config.broker].values, name="Ind. Invest"))
    fig1.add_trace(go.Scatter(x=all.index, y=all[Config.retirement].values, name="Ret. Invest"))
    fig1.add_trace(go.Scatter(x=all.index, y=all['Total'].values, name="Total"))
    fig1.update_layout(title_text="Asset Growth")
    yplot1 = image_path + key + '1.png'
    fig1.write_image(yplot1)
    add_plot(growth_plots, fig1)

    # Monthly Rate1
    all['Liquid' + key] = all[Config.balance].pct_change()*100
    all[Config.broker + key] = all[Config.broker].pct_change()*100
    all[Config.retirement + key] = all[Config.retirement].pct_change()*100
    all['Total' + key] = all['Total'].pct_change()
    fig1 = go.Figure()
    skip_rows = 2
    all = all[skip_rows:]
    lmean = all['Liquid' + key].values.mean()
    bmean = all[Config.broker + key].values.mean()
    rmean = all[Config.retirement + key].values.mean()
    fig1.add_trace(go.Bar(x=all.index, y=all['Liquid' + key].values, name="Liquid Growth"))
    fig1.add_trace(go.Scatter(x=all.index, y=[lmean for x in all.index], name="Liquid MeanGr"))
    fig1.add_trace(go.Bar(x=all.index, y=all[Config.broker+key].values, name="Ind. Growth"))
    fig1.add_trace(go.Scatter(x=all.index, y=[bmean for x in all.index], name="Ind. MeanGr"))
    fig1.add_trace(go.Bar(x=all.index, y=all[Config.retirement+key].values, name="Ret. Growth"))
    fig1.add_trace(go.Scatter(x=all.index, y=[rmean for x in all.index], name="Ret. MeanGr"))

    monthly_gr_rate = round((lmean + bmean + rmean) / 3,2)
    fig1.add_trace(go.Bar(x=all.index, y=all['Total'+key].values, name="Total Growth"))
    fig1.update_layout(title_text="Monthly Asset Growth Rate : "+ str(monthly_gr_rate)+"%")
    yplot1 = image_path + key + '2.png'
    fig1.write_image(yplot1)
    add_plot(growth_plots, fig1)

    start_liq = all[Config.balance].head(1).values[0]
    end_liq = all[Config.balance].tail(1).values[0]
    start_broker = all[Config.broker].head(1).values[0]
    end_broker = all[Config.broker].tail(1).values[0]
    start_ret = all[Config.retirement].head(1).values[0]
    end_ret = all[Config.retirement].tail(1).values[0]
    start_net = all["Total"].head(1).values[0]
    end_net = all["Total"].tail(1).values[0]
    util = Util()
    years = len(liq.groupby(Config.year_col).groups)
    months = all.shape[0]
    liq_cagr = util.cgr(start_liq, end_liq, years)
    bro_cagr = util.cgr(start_broker, end_broker, years)
    ret_cagr = util.cgr(start_ret, end_ret, years)
    net_cagr = util.cgr(start_net, end_net, years)
    liq_cmgr = util.cgr(start_liq, end_liq, months)
    bro_cmgr = util.cgr(start_broker, end_broker, months)
    ret_cmgr = util.cgr(start_ret, end_ret, months)
    net_cmgr = util.cgr(start_net, end_net, months)
    cagr={"Liquid": liq_cagr, "Ind.": bro_cagr, "Ret": ret_cagr, "Net": net_cagr}
    cmgr={"Liquid": liq_cmgr, "Ind.": bro_cmgr, "Ret": ret_cmgr, "Net": net_cmgr}
    growth_text.append("Monthly Mean Growth Rate: "+str(monthly_gr_rate)+"%")
    for ass in cagr:
        growth_text.append(ass+" CAGR : " + str(cagr[ass]) + "% :: \t"+ ass + " CMGR : " + str(cmgr[ass]) + "%\n\n")



asset_networth_summary, assets = do_asset_networth(balance_df, bro, ret, key, 'ass', use_latest_prices)
liab_networth_summary, liabs = do_liability_networth(key, 'lia')
do_growth_analysis(balance_df, bro, ret, "growth")

net = pd.concat([assets, liabs], axis=0)
net.loc["Total"] = round(net.values.sum(),2)
net.loc["Invested"] = round(assets[assets.index!='Liquid'].values.sum(),2)
net.loc["Debt"] = liabs.values.sum()
net.loc[Config.date_col] = Config.strtoday

networth_top = []
total_NW = "${:,.2f}".format(net.loc["Total"].values[0])
networth_top.append("Total Net Worth :   " + "${:,.2f}".format(net.loc["Total"].values[0]))
networth_top += asset_networth_summary
networth_top += liab_networth_summary

profile = Config.history_path+Config.net_worth_profile
if os.path.isfile(profile):
    history = pd.read_csv(profile)
    net = net.T[Config.net_worth_fields]
    if history.tail(1)[Config.date_col].values[0] != net[Config.date_col].values[0] or history.tail(1)['Total'].values[0] != net['Total'].values[0]:
        net.reset_index(inplace=True)
        net['index']=1+int(history.tail(1).Index)
        net.to_csv(profile, mode='a', header=False, index=False)
else:
    net.T[Config.net_worth_fields].to_csv(profile)






def serve_layout():
    layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H1(children=key + " : " + total_NW), className="mb-2"),
                dbc.Col(html.H4(children='Period: '+period), className="mb-2", width='auto'),
                dbc.Col(
                    html.Div(
                        daq.BooleanSwitch(id='use-latest-switch', on=False)
                    )
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=[html.H4("Summary")] + [html.H5(txt) for txt in networth_top]
                             , body=True, color="dark", outline=True, style={"width": 600}, className="text-left"),
                width='auto'),
                dbc.Col(dbc.Card(children=
                         [dcc.Graph(figure=top_plots[i][Config.plot_word], id='plot_' + str(i)) for i in top_plots.keys()]
                         , body=True, color="dark", outline=True, style={"width": 800}, className="text-center"),
                        width='auto')
            ],justify='center'),
            dbc.Row([
                dbc.Col(dbc.Card(children=
                            [html.H5(x) for x in growth_text]+
                             [dcc.Graph(figure=growth_plots[i][Config.plot_word], id='plot_' + str(i)) for i in growth_plots.keys()]
                             , body=True, color="dark", outline=True, style={"width": 1200}, className="text-center"),
                    width='auto')
            ], justify='center'),


        ])
    ])

    return layout

layout = serve_layout()
# layout = html.Div([
#         dbc.Container([
#             dbc.Row([
#                 dbc.Col(html.H1(children=key + " : " + total_NW), className="mb-2"),
#                 dbc.Col(html.H4(children='Period: '+period), className="mb-2", width='auto'),
#                 dbc.Col(
#                     html.Div(
#                         daq.BooleanSwitch(id='use-latest-switch1', on=False)
#                     )
#                 )
#             ]),
#             ])
#         ])



@callback(
    Output('container', 'children'),
    [Input('use-latest-switch', 'on')]
)
def update_switch(on):
    use_latest_prices = on
    print(f'The switch is {on}.')
    asset_networth_summary, assets = do_asset_networth(balance_df, bro, ret, key, 'ass', use_latest_prices)
    layout = serve_layout()
    return layout

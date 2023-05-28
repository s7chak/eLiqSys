import os
import sys

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import yfinance as yf
import yahoo_fin.stock_info as si
from dash import dcc, dash_table, Input, Output, callback, State
from dash import html
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config import Config
from ops.opapp import Util, Cleaner, InvestorFunctions
import dash

from ops.opscraper import YahooScraper, DataromaScraper, MarketBeatScraper

dash.register_page(__name__, path='/investisight')


use_inv_latest_prices = False
print("InvestiSight home page opened.")
file = Config.wd_path+Config.inv_final_data_file
if not os.path.isfile(file):
    util = Util()
    data_status = util.read_investment_data()
    if 'Success' not in data_status:
        print(Config.line_string)
df = pd.read_csv(file, sep=',')
period = df[Config.date_col].min()+" - " + df[Config.date_col].max()

ret = df[df[Config.acc_type]==Config.retirement]
bro = df[df[Config.acc_type]==Config.broker]

cleaner = Cleaner()
sign = 1
key = Config.investisight_portfolio
report_data = {key:{}}
report_data[key][Config.appendix_title]={}
top_1_plots={}
top_s_plots={}
top_i_plots={}
top_3_plots={}
top_plots={}
mid_plots={}
catplots={}
plot_num=0
txt_data={}
image_path = Config.today_folder+Config.plot_path_prefix
common_stocks = []
etfs = []
indices = []

print("File read for "+key+" analysis")
if not os.path.exists(Config.today_folder):
    os.mkdir((Config.today_folder))
if not os.path.exists(Config.today_folder+Config.plot_path_prefix):
    os.mkdir(Config.today_folder+Config.plot_path_prefix)
    print("Reporting folder created for today:"+Config.today_folder)


util = Util()


def dataroma_analysis(latest=False):
    dataroma_scraper = DataromaScraper()
    stocks, dr_res = dataroma_scraper.scanDataroma(latest)

    return dr_res


def yahoo_analysis(latest=False):
    link_map = {
        'Undervalued Growth': "https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks",
        'High-Yield Bond': "https://finance.yahoo.com/screener/predefined/high_yield_bond",
        'Tech Growth': "https://finance.yahoo.com/screener/predefined/growth_technology_stocks",
        'Undervalued LargeCaps': "https://finance.yahoo.com/screener/predefined/undervalued_large_caps",
        'Foreign Fund Index': "https://finance.yahoo.com/screener/predefined/conservative_foreign_funds",
        'Large Growth Funds': "https://finance.yahoo.com/screener/predefined/solid_large_growth_funds",
        'Small Growth Gainer': "https://finance.yahoo.com/screener/predefined/small_cap_gainers",
        'Mid Growth Funds': "https://finance.yahoo.com/screener/predefined/solid_midcap_growth_funds",
        'High Dividend Stocks': "https://finance.yahoo.com/screener/predefined/a656141d-407f-48dd-a5b4-60f49c1753dc?.tsrc=fin-srch"
        #     'Strong Undervalued Stocks': "https://finance.yahoo.com/screener/predefined/strong_undervalued_stocks" # free trial
        #     'Strong Buy Picks' : "https://finance.yahoo.com/screener#:~:text=Analyst%20Strong%20Buy%20Stocks", # free trial
        #     'Most Buy Stocks' : "https://finance.yahoo.com/screener/predefined/most_institutionally_bought_large_cap_stocks", # free trial
        #     'Most Buy Hedge': "https://finance.yahoo.com/screener#:~:text=all%2050%2B%20matches-,Stocks%20Most%20Bought%20by%20Hedge%20Funds,-Stocks",
        #     'Most Buy Institutional': "https://finance.yahoo.com/screener/predefined/stocks_with_most_institutional_buyers",
        #     'Undervalued Wide-Moat Stocks': "https://finance.yahoo.com/screener#:~:text=Undervalued%20Wide%2DMoat%20Stocks"

    }
    # ysutil = YahooScraper()
    yahoo_stocklist_elements = ['//*[@id="scr-res-table"]/div[1]/table/tbody/tr{num}',
                                '//*[@id="screener-results"]/div[1]/div[2]/div[1]/table/tbody/tr{num}',
                                '//*[@id="Col1-0-WatchlistDetail-Proxy"]/div/section[3]/div/div/table/tbody/tr{num}']
    ysutil = YahooScraper()
    result = ysutil.scanYahooLists(link_map, yahoo_stocklist_elements, latest)

    return result


def marketbeat_analysis(latest=False):
    link_map = {
        'Large Caps': "https://www.marketbeat.com/stocks/",
        'All Ratings': "https://www.marketbeat.com/ratings/",
    }
    marketbeat_stocklist_elements = ['//*[@id="form1"]/div[5]/div/table/tbody/tr{num}', '//*[@id="cphPrimaryContent_pnlUpdate"]/div[3]/div/table/tbody/tr{num}']
    mbsutil = MarketBeatScraper()
    result = mbsutil.scanMarketBeatLists(link_map, marketbeat_stocklist_elements, latest)

    return result


def do_stock_analysis(stock_list):
    if not os.path.isfile(Config.stock_scoring_file):
        top_data = pd.DataFrame()
        for tkr in stock_list:
            try:
                data = si.get_quote_table(tkr)
            except:
                print(tkr+" not found.")
                continue
            data[Config.symbol] = tkr
            top_data = pd.concat([top_data,pd.DataFrame([data])], ignore_index=True)
            if data:
                print(tkr + " found")
        scoring_metrics = list(Config.score_sheet.keys())
        top_data.rename(columns={"PE Ratio (TTM)": scoring_metrics[0],
                                 "EPS (TTM)": scoring_metrics[1],
                                 "Forward Dividend & Yield": scoring_metrics[2],
                                 "Beta (5Y Monthly)": scoring_metrics[3]}, inplace=True)
        top_data[scoring_metrics[1]] = round(top_data[scoring_metrics[1]] / top_data["Quote Price"],5)  #Earnings / Price Ratio
        top_data.to_csv(Config.stock_scoring_file)
        scoring = top_data
    else:
        scoring = pd.read_csv(Config.stock_scoring_file)
    my_score = calculate_scores(scoring, stock_list)
    return my_score


def calculate_scores(scoring, stock_list):
    result = pd.DataFrame(columns=[Config.symbol, Config.my_score])
    result[Config.symbol] = pd.Series(stock_list)
    scoring['DivYield'] = scoring['DivYield'].apply(lambda x: str(x).split("(")[1].split(")")[0].replace('%','').strip() if '%' in str(x) else 0)
    for metric in Config.score_sheet:
        scoring[metric+"_score"] = 0
    for i, row in scoring.iterrows():
        scoring['TargetGrSpread'] = (scoring['1y Target Est'] - scoring['Quote Price']) / scoring['Quote Price']
        scoring['TargetGrSpread'].fillna(0, inplace=True)
        for metric in Config.score_sheet:
            add = 1
            det = Config.score_sheet[metric]
            if row[metric]:
                if det[0] == 'min':
                    if row[metric] < det[1]:
                        scoring.at[i, metric + "_score"] += add
                elif det[0] == 'max':
                    if float(row[metric]) > det[1]:
                        scoring.at[i, metric + "_score"] += add
    scoring['Total'] = scoring.loc[:, 'PE_score':'Beta_score'].sum(axis=1) + scoring['TargetGrSpread']
    score = scoring['Total']
    mn, mx = score.min(), score.max()
    score_scaled = (score - mn) / (mx - mn)
    result = pd.merge(scoring, result, how='left', on=Config.symbol)[[Config.symbol, 'TargetGrSpread', Config.my_score]]
    result[Config.my_score] = score_scaled.values

    return result


def aggregate_scores(y, mk, d, my):
    y_factor = 0.5
    m_factor = 0.5
    d_factor = 1
    my_factor = 1
    result = pd.concat([y,mk,d,my],axis=0)
    result.fillna(0, inplace=True)
    result = result.groupby('Symbol')[['Yahoo_Score', 'MarketBeat_Score', 'Dataroma_Total_Score', Config.my_score]].sum().reset_index()
    result['Total'] = y_factor*result['Yahoo_Score']+ m_factor*result['MarketBeat_Score'] + d_factor*result['Dataroma_Total_Score'] + my_factor*result[Config.my_score]
    top_scores = pd.DataFrame(result.sort_values('Total', ascending=False))
    top_scores.to_csv(Config.history_path+Config.strtoday+'_top_scores.csv')
    return top_scores

take_top = 20
dataroma = dataroma_analysis().head(take_top)
yahoo = yahoo_analysis().head(take_top)
marketbeat = marketbeat_analysis().head(take_top)

stock_list = list(pd.concat([dataroma, yahoo, marketbeat], axis=0)[Config.symbol].unique())
my_scores = do_stock_analysis(stock_list)

all_scores = aggregate_scores(yahoo, marketbeat, dataroma, my_scores)


def add_plot(pts, fig):
    plot_num = len(pts.keys())
    pts[plot_num]={}
    pts[plot_num][Config.plot_word] = fig

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

def find_price(fund):
    mkb_util = MarketBeatScraper()
    return mkb_util.get_price_symbol(fund)

def get_fund_current_price(fund):
    try:
        if '-' in fund:
            fund = fund.split(' ')[0]
        ticker = yf.Ticker(fund)
        fund_df = ticker.history(period='2d')
        if fund_df.empty:
            last = find_price(fund)
        else:
            last = fund_df['Close'].tail(1).values[0]
        return last

    except:
        print("Fund data not found: "+fund+' - '+str(sys.exc_info()[1]))
        return None

def get_fund_daily_history(fund):
    try:
        if '-' in fund:
            fund = fund.split(' ')[0]
        ticker = yf.Ticker(fund)
        if not os.path.isfile(Config.history_path+fund+'.csv'):
            fund_df = ticker.history(period="2mo", interval='1d')
            fund_df.rename(columns={'Close': Config.amount}, inplace=True)
            if fund_df.empty:
                return None
            fund_df[Config.trade_date] = fund_df.index.strftime('%Y-%m-%d')
        else:
            fund_df = pd.read_csv(Config.history_path+fund+'.csv')
    except:
        print('Fund not found: ' + fund)
        return None
    fund_df = fund_df[[Config.trade_date,Config.amount]]
    fund_df.to_csv(Config.history_path+fund+'.csv')
    return fund_df.set_index(Config.trade_date)


def get_fund_history(fund):
    try:
        if '-' in fund:
            fund = fund.split(' ')[0]
        ticker = yf.Ticker(fund)
        if not os.path.isfile(Config.history_path+fund+'.csv'):
            fund_df = ticker.history(period="max", start=df[Config.date_col].min())
            fund_df.rename(columns={'Close': Config.amount}, inplace=True)
            if fund_df.empty:
                if 'NLOK' in fund:
                    fund_df = yf.Ticker(fund+'.VI').history(period='max')
                    return fund_df

                return None
            fund_df[Config.ym_col] = fund_df.index.strftime('%Y-%m')
        else:
            fund_df = pd.read_csv(Config.history_path+fund+'.csv')
    except:
        print('Fund not found: ' + fund)
        return None
    fund_df = pd.DataFrame(fund_df.groupby(Config.ym_col)[Config.amount].mean())
    fund_df.to_csv(Config.history_path+fund+'.csv')
    fund_df = fund_df[fund_df.index > df[Config.date_col].min()]
    return fund_df




def do_pie_analysis(b, r, key, suf):
    cost_brok_total = float(b[Config.amount].sum())
    cost_k401_total = float(r[Config.my_spend].sum())
    val_brok_total = float(b[Config.current_amount].sum())
    val_k401_total = float(r[Config.current_amount].sum())
    b.set_index(Config.symbol, inplace=True)
    r.set_index(Config.symbol.split('-')[0].strip(), inplace=True)
    # pie categories
    fig1 = go.Figure()
    fig1.add_trace(go.Pie(labels=b.index, values=b[Config.current_amount], hole=0.3, name='Value'))
    fig2 = go.Figure()
    fig2.add_trace(go.Pie(labels=b.index, values=b[Config.amount], hole=0.3, name='Cost'))
    fig1.update_layout(title_text="Liquid Value", width=700, annotations=[dict(text="${:,.2f}".format(val_brok_total), x=0.5, y=0.5, font_size=12, showarrow=False)])
    fig2.update_layout(title_text='Liquid Cost',width=700, annotations=[dict(text="${:,.2f}".format(cost_brok_total), x=0.5, y=0.5, font_size=12, showarrow=False)])
    index_labels = [x.split('-')[0].strip() if '-' in x else x.split('(')[0].strip() for x in r.index]
    fig3 = go.Figure()
    fig3.add_trace(go.Pie(labels=index_labels, values=r[Config.current_amount], hole=0.3, name='Value'))
    fig4 = go.Figure()
    fig4.add_trace(go.Pie(labels=index_labels, values=r[Config.my_spend], hole=0.3, name='Cost'))
    fig3.update_layout(title_text='Ret Value',width=800,annotations=[dict(text="${:,.2f}".format(val_k401_total), x=0.5, y=0.5, font_size=12, showarrow=False)])
    fig4.update_layout(title_text='Ret Cost',width=800,annotations=[dict(text="${:,.2f}".format(cost_k401_total), x=0.5, y=0.5, font_size=12, showarrow=False)])
    yplot1 = image_path + key + suf + '-pie-1.png'
    yplot2 = image_path + key + suf + '-pie-2.png'
    yplot3 = image_path + key + suf + '-pie-3.png'
    yplot4 = image_path + key + suf + '-pie-4.png'
    fig1.write_image(yplot1)
    fig2.write_image(yplot2)
    fig2.write_image(yplot3)
    fig2.write_image(yplot4)
    add_plot(top_s_plots, fig1)
    add_plot(top_s_plots, fig2)
    add_plot(top_i_plots, fig3)
    add_plot(top_i_plots, fig4)


def do_monthly_analysis(b, r, key):
    # bar months
    fig1 = go.Figure()
    xaxis = b[Config.ym_col].values
    b = b[(b[Config.qty].notna()) & (b[Config.amount]>0)]
    symbols = b[Config.symbol].unique()
    sum_monthly_investment = b.groupby(Config.ym_col)[Config.amount].sum()
    overall_monthly_mean = sum_monthly_investment.sum() / len(xaxis)
    for sym in symbols:
        buy_hist = b[b[Config.symbol]==sym]
        buy_hist = buy_hist.groupby(Config.ym_col)[Config.amount].sum()
        fig1.add_trace(go.Bar(x=buy_hist.index, y=buy_hist.values, name=sym))
    fig1.add_trace(go.Scatter(x=xaxis, y=[overall_monthly_mean for i in xaxis], name='MonthlyMean'))
    fig2 = go.Figure()
    r = r[(r[Config.qty].notna()) & (r[Config.amount] > 0) & (r[Config.description].str.contains('Employee'))]
    symbols = r[Config.symbol].unique()
    sum_monthly_investment = r.groupby(Config.ym_col)[Config.amount].sum()
    overall_monthly_mean = sum_monthly_investment.mean()
    for sym in symbols:
        buy_hist = r[r[Config.symbol] == sym]
        buy_hist = buy_hist.groupby(Config.ym_col)[Config.amount].sum()
        sym_name = sym.split('-')[0] if '-' in sym else sym.split('k')[0]
        fig2.add_trace(go.Bar(x=buy_hist.index, y=buy_hist.values, name=sym_name))
    fig2.add_trace(go.Scatter(x=xaxis,y=[overall_monthly_mean for i in xaxis], name='MonthlyMean'))
    yplot1 = image_path + key + '-stocks.png'
    yplot2 = image_path + key + '-index.png'
    fig1.write_image(yplot1)
    fig2.write_image(yplot2)
    add_plot(top_s_plots, fig1)
    add_plot(top_i_plots, fig2)


def do_yearly_analysis(b, r, key):
    # bar months
    fig1 = go.Figure()
    b = b[(b[Config.qty].notna()) & (b[Config.amount] > 0)]
    b[Config.year_col] = b[Config.date_col].str[:4]
    years = b[Config.year_col].unique()
    mean_yearly_investment = b.groupby(Config.year_col)[Config.amount].sum()
    overall_mean = mean_yearly_investment.mean()
    b = b.groupby(Config.year_col)[Config.amount].sum()
    fig1.add_trace(go.Bar(x=b.index, y=b, name='Ind. Yearly Investment'))
    fig1.add_trace(go.Scatter(x=years, y=[overall_mean for x in years], name='OverallYearlyMean'))
    fig2 = go.Figure()
    r = r[(r[Config.qty].notna()) & (r[Config.amount] > 0) & (r[Config.description].str.contains('Employee'))]
    r[Config.year_col] = r[Config.date_col].str[:4]
    mean_yearly_investment = r.groupby(Config.year_col)[Config.amount].sum()
    overall_mean = mean_yearly_investment.mean()
    r = r.groupby(Config.year_col)[Config.amount].sum()
    fig2.add_trace(go.Bar(x=r.index, y=r, name='Ret. Yearly Investment'))
    fig2.add_trace(go.Scatter(x=r.index, y=[overall_mean for x in r.index], name='OverallYearly'))
    yplot1 = image_path + key + '-stocks.png'
    yplot2 = image_path + key + '-index.png'
    fig1.write_image(yplot1)
    fig2.write_image(yplot2)
    add_plot(top_s_plots, fig1)
    add_plot(top_i_plots, fig2)

def get_fund_last_price(sym, is_stock):
    cur_price_file = Config.history_path + 'stck_current_price_list.csv' if is_stock else Config.history_path +  'indx_current_price_list.csv'
    if os.path.isfile(cur_price_file):
        prices = pd.read_csv(cur_price_file)
        latest = prices[prices[Config.symbol]==sym][Config.price].values[0]
        return latest
    return



def do_portfolio_table(bro, ret, key):
    # all = combined.groupby(Config.symbol)[Config.amount].sum().sort_values(ascending=False)
    investor = InvestorFunctions(use_inv_latest_prices)
    brokerage, retirement = investor.analyze_investment_portfolio(bro, ret)
    elems, data = investor.sectype_analysis(brokerage, retirement)

    bro_gain = brokerage[Config.fund_gain].sum()
    net_bro = brokerage['CurrentAmount'].sum()
    ret_gain = retirement[Config.fund_gain].sum()
    net_ret = retirement['CurrentAmount'].sum()



    today_entry = pd.DataFrame(index=[1],columns=[Config.date_col] + Config.rec_columns)
    # ['NetPortfolio', 'NetLiquid', 'NetRetirement', 'TotalGain', 'TotalLiquidGain', 'TotalRetirementGain', 'NumFunds',
     # 'NumStocks', 'NumETFs', 'NumIndex']
    today_entry[Config.date_col] = df[Config.date_col].max()
    today_entry['NetPortfolio'] = net_bro + net_ret
    today_entry['NetLiquid'] = net_bro
    today_entry['NetRetirement'] = net_ret
    today_entry['TotalGain'] = bro_gain + ret_gain
    today_entry['TotalGain%'] = (bro_gain + ret_gain)*100 / (net_bro + net_ret)
    today_entry['TotalLiquidGain'] = bro_gain
    today_entry['TotalLiquidGain%'] = bro_gain*100 / net_bro
    today_entry['TotalRetirementGain'] = ret_gain
    today_entry['TotalRetirementGain%'] = ret_gain*100 / net_ret
    total_funds = elems[Config.stock][0] + elems[Config.etf][0] + elems[Config.index][0] + elems[Config.bond][0] + elems[Config.real_estate][0]
    today_entry['NumFunds'] = total_funds
    today_entry['NumStocks'] = elems[Config.stock][0]
    today_entry['NumETFs'] = elems[Config.etf][0]
    today_entry['NumIndex'] = elems[Config.index][0]
    today_entry['NumBonds'] = elems[Config.bond][0]
    today_entry['NumRealEst'] = elems[Config.real_estate][0]
    today_entry['ListStocks'] = "|".join(elems[Config.stock][1])
    today_entry['ListETF'] = "|".join(elems[Config.etf][1])
    today_entry['ListIndex'] = "|".join(elems[Config.index][1])
    today_entry['ListBonds'] = "|".join(elems[Config.bond][1])
    today_entry['ListRealEst'] = "|".join(elems[Config.real_estate][1])

    profile = Config.history_path + Config.portfolio_history_file

    if not os.path.isfile(profile):
        today_entry.to_csv(profile)
    else:
        history = pd.read_csv(profile, index_col=0)
        if history.tail(1)[Config.date_col].values[0] != today_entry[Config.date_col].values[0] or history.tail(1)['NetPortfolio'].values[0] != today_entry['NetPortfolio'].values[0]:
            today_entry.index = history.tail(1).index + 1
            today_entry.reset_index().to_csv(profile, mode='a', header=False, index=False)
            print("New entry in portfolio record: "+str(today_entry[Config.date_col].values[0]))

    return brokerage.reset_index(), retirement.reset_index(), bro_gain, ret_gain, net_bro, net_ret, elems, data






def do_categorize_portfolio(combined, key, suf):
    c_grouped = combined[combined[Config.my_spend].isna()].groupby(Config.sec_type_col)[Config.amount].sum()
    c_grouped = c_grouped.append(combined[~combined[Config.my_spend].isna()].groupby(Config.sec_type_col)[Config.my_spend].sum())
    v_grouped = combined.groupby(Config.sec_type_col)[Config.current_amount].sum()
    cost_total = c_grouped.sum()
    val_total = v_grouped.sum()
    gain = (val_total - cost_total)*100 / cost_total
    fig1 = go.Figure()
    fig1.add_trace(go.Pie(labels=v_grouped.index, values=v_grouped.values, hole=0.3))
    fig1.update_layout(
        title_text="Value Split:  Val = "+"${:,.2f}".format(val_total) + "  Cost = " + "${:,.2f}".format(cost_total) + " Gain: " + str(round(gain,2))+"%",
        annotations=[dict(text="${:,.2f}".format(val_total), x=0.5, y=0.5, font_size=10, showarrow=False)])
    yplot1 = image_path + key + suf + '-pie-1.png'
    fig1.write_image(yplot1)
    add_plot(top_1_plots, fig1)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=c_grouped.index, y=c_grouped.values))
    fig2.add_trace(go.Bar(x=v_grouped.index, y=v_grouped.values))
    yplot2 = image_path + key + suf + '-pie-2.png'
    fig2.write_image(yplot2)
    add_plot(top_1_plots, fig2)

def do_tornado_chart(b, r, key, suf):
    b = b.sort_values(Config.fund_gain, ascending=False)
    r = r.sort_values(Config.fund_gain, ascending=False)
    # pie categories
    fig1 = go.Figure()
    b_gain_total = b[Config.fund_gain].sum()
    b_gain_perct = round(b_gain_total*100 / b[Config.amount].sum(),2)
    fig1.add_trace(go.Bar(x=b.index.values, y=b[Config.fund_gain].values, name='Individual Gains'))
    fig1.update_layout(title_text="Total Ind. Gain: " + "${:,.2f}".format(b_gain_total) + "   ::   "+str(b_gain_perct)+"% ",
                       width=500,)
    r_gain_total = r[Config.fund_gain].sum()
    r_gain_perct = round(r_gain_total * 100 / r[Config.amount].sum(),2)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=r.index.values, y=r[Config.fund_gain].values, name='Retirement Gains'))
    fig2.update_layout(title_text="Total Ret. Gain: " + "${:,.2f}".format(r_gain_total) + "   ::   "+str(r_gain_perct)+"% ",
                       width=500,)
    yplot1 = image_path + key + suf + '-tor-1.png'
    yplot2 = image_path + key + suf + '-tor-2.png'
    fig1.write_image(yplot1)
    fig2.write_image(yplot2)
    add_plot(top_3_plots, fig1)
    add_plot(top_3_plots, fig2)



overall_gain = 0
br_portfolio, ret_portfolio, bro_gain, ret_gain, net_bro, net_ret, elems, sec_grouped = do_portfolio_table(bro, ret, key)
do_categorize_portfolio(sec_grouped, key, '-portcat')
do_pie_analysis(br_portfolio, ret_portfolio, key,'-split')
do_monthly_analysis(bro, ret, 'monthly')
do_yearly_analysis(bro, ret, 'Yearly')
overall_gain = bro_gain + ret_gain
net_portfolio = net_bro + net_ret
do_tornado_chart(br_portfolio, ret_portfolio, key, '-tornado')


site_list = Config.list_sites
bro_list = bro[bro[Config.qty].notna() & bro[Config.amount]>0][Config.symbol].unique()
ret_list = ret[ret[Config.qty].notna()][Config.symbol].unique()
stck_dropdown = dcc.Dropdown(
                options=Config.stck_watch_list,
                value= Config.stck_watch_list,
                multi=True,
                id='stck-dropdown-id',
                placeholder="Select Stocks",
                style={'width': '100%', 'color':'black'}
            )

bro_dropdown = dcc.Dropdown(
                options=bro_list,
                value= bro_list,
                multi=True,
                id='bro-dropdown-id',
                placeholder="Select funds",
                style={'width': '100%', 'color':'black'}
            )

ret_dropdown = dcc.Dropdown(
                options=ret_list,
                value= ret_list,
                multi=True,
                id='ret-dropdown-id',
                placeholder="Select funds",
                style={'width': '100%', 'color':'black'}
            )


b_display = br_portfolio[['Percentage','Cost_Basis','Current_Price','Gain %']]
r_display = ret_portfolio[['Value_Percentage','Cost_Basis','Current_Price','Gain %',Config.my_fund_gain_perct, 'MyCost_Percentage','Purchase_Percentage']]

def serve_layout():
    portfolio_modal_body = html.Div([
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col(html.H4(children='Total Investment'), className="mb-2"),
                                    dbc.Col(html.H5(children='Gain:  '+str(round(overall_gain*100/net_portfolio,2))+"%"), className="mb-2"),
                                ], style={'width': "auto"}),
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(children=
                                                 [dcc.Graph(figure=top_1_plots[i][Config.plot_word],
                                                            id='plot_top_1_' + str(i)) for i in top_1_plots.keys()]
                                                 , body=True, color="dark", outline=True,
                                                 style={"width": 700, "height": 500},
                                                 className="text-center"), width="auto",
                                    ),
                                ], justify='center'),
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(children=[
                                            html.H4("Total Worth :  " + "${:,.2f}".format(net_portfolio)),
                                            html.H5("Total Liquid Worth :  " + "${:,.2f}".format(net_bro)),
                                            html.H5("Total Retirement Worth :  " + "${:,.2f}".format(net_ret)),
                                        ], style={'width': 500}), width='auto',
                                    ),
                                    dbc.Col(
                                        dbc.Card(children=[
                                            html.H4("Total Gain :  " + "${:,.2f}".format(overall_gain)),
                                            html.H5("Total Liquid Gain :  " + "${:,.2f}".format(bro_gain)),
                                            html.H5("Total Retirement Gain :  " + "${:,.2f}".format(ret_gain)),
                                        ], style={'width': 500}), width='auto',
                                    )
                                ], justify='center'),

                            ])
        ], style={"justify":"center"})
    stck_portfolio_modal_body = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H4(children='Stocks'), className="mb-2"),
                dbc.Col(html.H5(children="Total worth: "+"${:,.2f}".format(net_bro)), className="mb-2"),
                dbc.Col(html.H5(children="Gain:  " + str(round(bro_gain * 100 / net_bro,2))+"%"), className="mb-2"),
            ], style={'width': "auto"}),
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=
                             [dcc.Graph(figure=top_s_plots[0][Config.plot_word], id='plot_bro_split_value')] +
                             [
                                dash_table.DataTable(
                                    b_display.reset_index().to_dict('records'),
                                    [{"name": i, "id": i} for i in b_display.reset_index().columns],
                                    style_cell={'textAlign': 'center'},
                                    style_header={
                                        'backgroundColor': 'black',
                                        'fontWeight': 'bold'
                                    },
                                    style_data={
                                        'color': 'primary',
                                        'backgroundColor': 'black'
                                    },
                                )
                            ]+
                             [dcc.Graph(figure=top_s_plots[1][Config.plot_word], id='plot_bro_split_cost')] +
                             [dcc.Graph(figure=top_s_plots[2][Config.plot_word], id='plot_bro_monthly')] +
                             [dcc.Graph(figure=top_3_plots[0][Config.plot_word], id='plot_bro_gains')]
                             , body=True, color="dark", outline=True,
                             style={"width": 500, "height": 500},
                             className="text-center"), width="auto",
                ),
            ], justify='center'),

        ])
    ])
    index_portfolio_modal_body = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H4(children='Indices'), className="mb-2"),
                dbc.Col(html.H5(children="Total worth: " + "${:,.2f}".format(net_ret)), className="mb-2"),
                dbc.Col(html.H5(children="Gain:  " + str(round(ret_gain*100/net_ret,2))+"%"), className="mb-2"),
            ], style={'width': "auto"}),
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=
                             [dcc.Graph(figure=top_i_plots[0][Config.plot_word], id='plot_ret_split_value')]+
                             [
                                 dash_table.DataTable(r_display.reset_index().to_dict('records'),
                                                      [{"name": i, "id": i} for i in r_display.reset_index().columns],
                                                      style_cell={'textAlign': 'center'},
                                                      style_header={
                                                          'backgroundColor': 'black',
                                                          'color': 'primary',
                                                          'fontWeight': 'bold'
                                                      },
                                                      style_data={
                                                          'color': 'primary',
                                                          'backgroundColor': 'black'
                                                      },
                                                      )
                             ]+
                             [dcc.Graph(figure=top_i_plots[1][Config.plot_word], id='plot_ret_split_cost')] +
                             [dcc.Graph(figure=top_i_plots[2][Config.plot_word], id='plot_ret_monthly')] +
                              [dcc.Graph(figure=top_3_plots[1][Config.plot_word], id='plot_ret_gains')]
                             , body=True, color="dark", outline=True,
                             style={"width": 500, "height": 500},
                             className="text-center"), width="auto",
                ),
            ], justify='center'),

        ])
    ])
    layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H1(children="InvestiSight"), className="mb-2"),
                dbc.Col(html.H4(children='Period: '+period), className="mb-2, text-right", width='auto')
            ], style={'width':"auto"}),
            html.Div(
                [
                    dbc.Button("Portfolio", id="port_open"),
                    dbc.Button("Stock Portfolio", id="stck_open"),
                    dbc.Button("Index Portfolio", id="indx_open"),

                ], style={'width': 600},
            ),
            dbc.Row([
                dbc.Col(
                    stck_dropdown
                    , style={"width": 600}, className="text-center"),
                dcc.Input(id="stck-text-list", type="text", placeholder="Enter stock symbols", style={'marginRight': '10px'}),
                dbc.Card(children=[
                                      html.H5("Markets")
                                  ] +
                                  [
                                      dcc.Graph(id='stck-line-plot_1'),
                                  ])
            ]),
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        all_scores.to_dict('records'),
                        [{"name": i, "id": i} for i in all_scores.columns],
                        style_cell={'textAlign': 'center'},
                        style_header={
                            'backgroundColor': 'black',
                            'fontWeight': 'bold'
                        },
                        style_data={
                            # 'color': 'primary',
                            'backgroundColor': 'black'
                        },
                    ), width='auto',
                ),
            ], justify='center'),

            dbc.Row([
                dbc.Col(
                    bro_dropdown
                    , style={"width": 600}, className="text-center"),
                dbc.Col(
                    ret_dropdown
                    , style={"width": 600}, className="text-center"),
            ]),
            dbc.Row([
                dbc.Card(children=[
                                      html.H5("Individual Monthly vs Actual")
                                  ] +
                                  [
                                      dcc.Graph(id='bro-line-plot_1'),
                                  ])
            ]),
            dbc.Row([
                dbc.Card(children=[
                                      html.H5("Retirement Monthly vs Actual")
                                  ] +
                                  [
                                      dcc.Graph(id='ret-line-plot_1'),
                                  ])
            ]),
            dbc.Offcanvas(
                portfolio_modal_body,
                id="portfolio-canvas",
                title="Portfolio",
                is_open=False,
                placement='bottom',
                style={"width":1800, "height": 1000, "justify":"center"},
            ),
            dbc.Offcanvas(
                stck_portfolio_modal_body,
                id="stck_portfolio",
                title="Stock Portfolio",
                is_open=False,
                placement='start',
                style={"width": 1100},
            ),
            dbc.Offcanvas(
                index_portfolio_modal_body,
                id="indx_portfolio",
                title="Index Portfolio",
                is_open=False,
                placement='end',
                style={"width": 1100},
            ),
            html.Div(id='data_status', className='text-center'),
        ])
    ])


    return layout


layout = serve_layout()



@callback(
    Output("portfolio-canvas", "is_open"),
    Input("port_open", "n_clicks"),
    [State("portfolio-canvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


@callback(
    Output("stck_portfolio", "is_open"),
    Input("stck_open", "n_clicks"),
    [State("stck_portfolio", "is_open")],
)
def toggle_stckoffcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@callback(
    Output("indx_portfolio", "is_open"),
    Input("indx_open", "n_clicks"),
    [State("indx_portfolio", "is_open")],
)
def toggle_retoffcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


def find_average_price(df):
    df = df[df[Config.qty].notna()]
    price_sum = df[Config.amount].sum()
    qty_sum = df[Config.qty].sum()

    return round(price_sum / qty_sum , 2)


stock_lines = []
@callback(
    Output(component_id='stck-line-plot_1',component_property='figure'),
    [Input(component_id='stck-dropdown-id', component_property='value'),
     Input(component_id='stck-text-list', component_property='value')]
)
def stck_lines(stcks, inp_list):
    if inp_list is not None and inp_list!='':
        stcks += inp_list.split(',')
    if len(stcks) > 0:
        brodf = bro[bro[Config.symbol].isin(stcks)]
        stck_traces = []
        fig = go.Figure()
        for c in stcks:
            hist = get_fund_daily_history(c)
            if hist is not None and Config.amount in hist.columns:
                fund_df = brodf[brodf[Config.symbol] == c]
                stck_traces.append(go.Scatter(x=hist.index, y=hist[Config.amount].values, name=c))
                if c in brodf[Config.symbol].unique():
                    avg = find_average_price(fund_df)
                    if avg is not None:
                        stck_traces.append(go.Scatter(x=hist.index, y=[avg for x in hist.index], name=c+'-my-avg'))
        fig.add_traces(stck_traces)
        return fig
    else:
        return go.Figure()



@callback(
    Output(component_id='bro-line-plot_1',component_property='figure'),
    [Input(component_id='bro-dropdown-id', component_property='value')]
)
def bro_current_analysis(bro_funds):
    if len(bro_funds) > 0:
        brodf = bro[bro[Config.symbol].isin(bro_funds)]
        bro_traces = []
        fig = go.Figure()
        avg_comparison = pd.DataFrame(index=bro_funds, columns=[Config.my_price, Config.price])
        current_price_stocks = pd.DataFrame(index=bro_funds, columns=[Config.price])
        current_price_stocks.index.name=Config.symbol
        for c in bro_funds:
            if use_inv_latest_prices:
                hist = get_fund_current_price(c)
                current_price_stocks.loc[c, Config.price] = round(hist, 2)
            else:
                hist = get_fund_last_price(c, True)
            if hist is not None:
                fund_df = brodf[brodf[Config.symbol]==c]
                avg = find_average_price(fund_df)
                avg_comparison.loc[c, Config.my_price] = round(avg,2)
                avg_comparison.loc[c, Config.price] = round(hist,2)
        if use_inv_latest_prices:
            current_price_stocks.reset_index().to_csv(Config.history_path + 'stck_current_price_list.csv')
        avg_comparison[Config.gain_col] = avg_comparison[Config.price] - avg_comparison[Config.my_price]
        avg_comparison[Config.color_col] = avg_comparison[Config.gain_col].apply(lambda x: "green" if x>0 else "red")
        avg_comparison = avg_comparison.sort_values(Config.gain_col, ascending=False)
        bro_traces.append(go.Bar(x=avg_comparison.index, y=avg_comparison[Config.my_price], name='My Avg Cost Price', marker=dict(color=avg_comparison[Config.color_col])))
        bro_traces.append(go.Bar(x=avg_comparison.index, y=avg_comparison[Config.price], name='Current Price'))
        fig.add_traces(bro_traces)
        return fig
    else:
        return go.Figure()


@callback(
    Output(component_id='ret-line-plot_1',component_property='figure'),
    [Input(component_id='ret-dropdown-id', component_property='value')]
)
def ret_current_analysis(ret_funds):
    if len(ret_funds) > 0:
        retdf = ret[ret[Config.symbol].isin(ret_funds)]
        ret_traces = []
        fig = go.Figure()
        avg_comparison = pd.DataFrame(index=ret_funds, columns=[Config.my_price, Config.price])
        current_price_stocks = pd.DataFrame(index=ret_funds, columns=[Config.price])
        current_price_stocks.index.name = Config.symbol

        for c in ret_funds:
            if use_inv_latest_prices:
                hist = get_fund_current_price(c)
                current_price_stocks.loc[c, Config.price] = round(hist, 2) if hist is not None else hist
            else:
                hist = get_fund_last_price(c, False)
            if hist is not None:
                fund_df = retdf[retdf[Config.symbol] == c]
                avg = find_average_price(fund_df)
                avg_comparison.loc[c, Config.my_price] = round(avg, 2)
                avg_comparison.loc[c, Config.price] = round(hist, 2)
        if use_inv_latest_prices:
            current_price_stocks.reset_index().to_csv(Config.history_path + 'indx_current_price_list.csv')
        avg_comparison[Config.gain_col] = avg_comparison[Config.price] - avg_comparison[Config.my_price]
        avg_comparison[Config.color_col] = avg_comparison[Config.gain_col].apply(lambda x: "green" if x > 0 else "red")
        avg_comparison = avg_comparison.sort_values(Config.gain_col, ascending=False)
        ret_traces.append(go.Bar(x=avg_comparison.index, y=avg_comparison[Config.my_price], name='My Avg Cost Price', marker=dict(color=avg_comparison[Config.color_col])))
        ret_traces.append(go.Bar(x=avg_comparison.index, y=avg_comparison[Config.price], name='Current Price'))
        fig.add_traces(ret_traces)
        return fig
    else:
        return go.Figure()






@callback(
    Output(component_id='broplot_1',component_property='figure'),
    [Input(component_id='bro-dropdown-id', component_property='value')]
)
def bro_monthly_analysis(bro_funds):
    if len(bro_funds) > 0:
        brodf = bro[bro[Config.symbol].isin(bro_funds)]
        bro_traces = []
        fig = go.Figure()
        for c in bro_funds:
            fund_df = brodf[brodf[Config.symbol]==c].groupby(Config.ym_col).sum()
            bro_traces.append(go.Bar(x=fund_df.index, y=fund_df[Config.amount].values, name=c))
        monthly = brodf.groupby(Config.ym_col).sum()
        mean = monthly[Config.amount].mean()
        bro_traces.append(go.Scatter(x=monthly.index, y=[mean for i in monthly.index], name='Monthly Mean'))
        fig.add_traces(bro_traces)
        return fig
    else:
        return go.Figure()


@callback(
    Output(component_id='retplot_1',component_property='figure'),
    [Input(component_id='ret-dropdown-id', component_property='value')]
)
def ret_monthly_analysis(ret_funds):
    if len(ret_funds) > 0:
        retdf = ret[ret[Config.symbol].isin(ret_funds)]
        ret_traces = []
        fig = go.Figure()
        for c in ret_funds:
            fund_df = retdf[retdf[Config.symbol]==c].groupby(Config.ym_col).sum()
            ret_traces.append(go.Bar(x=fund_df.index, y=fund_df[Config.amount].values, name=c.split(' ')[0] if '(' not in c else c.split('(')[0]))
        monthly = retdf.groupby(Config.ym_col).sum()
        mean = monthly[Config.amount].mean()
        ret_traces.append(go.Scatter(x=monthly.index, y=[mean for i in monthly.index], name='Monthly Mean'))
        fig.add_traces(ret_traces)
        return fig
    else:
        return go.Figure()


import os
import sys
from collections import Counter
# from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime
from os.path import exists
import yfinance as yf
import yahoo_fin.stock_info as si
import pandas as pd

from config import Config
from ops.opscraper import MarketBeatScraper


class Util():

    def save_dataframe(self, df:pd.DataFrame, path, name):
        df.to_csv(path+name)

    def get_average(self,d, cat, val):
        return pd.DataFrame(round(d.groupby(cat)[val].mean(), 2))

    def get_average_multi(self,d, cats, val):
        return pd.DataFrame(round(d.groupby(cats)[val].mean(), 2))

    def get_last_multi(self,d, cats, val):
        return pd.DataFrame(round(d.groupby(cats)[val].tail(1).reset_index(), 2))

    def get_sum(self,d, cat, val):
        return pd.DataFrame(round(d.groupby(cat)[val].sum(), 2))

    def get_sum_multi(self,d, cats, val):
        return pd.DataFrame(round(d.groupby(cats)[val].sum(), 2))

    def read_data(self):
        file = Config.wd_path + Config.data_csv
        if os.path.isfile(file):
            return "All data files read."
        print("Reading data files...")
        data_path = os.path.abspath(Config.data_path_prefix)
        file_list = [f for f in os.listdir(data_path) if ".csv" in f.lower()]
        print("Data Files found:")
        print("\n".join(file_list))
        print(Config.line_string)
        try:
            reader = Reader()
            data, status = reader.read_files(file_list)
            read_process_status = status
            if not os.path.exists(Config.wd_path):
                os.mkdir(Config.wd_path)
            data.to_csv(Config.wd_path + Config.data_csv)
            return read_process_status
        except:
            err = "Failure reading data: " + str(sys.exc_info()[1])
            print(err)
            return err
        return "Data not loaded."


    def read_investment_data(self):
        data_path = os.path.abspath(Config.data_path_prefix)
        file_list = [f for f in os.listdir(data_path) if ".csv" in f.lower()]
        print("Investment Data Files found:")
        print("\n".join(file_list))
        print(Config.line_string)
        try:
            reader = Reader()
            data, status = reader.read_inv_files(file_list)
            read_process_status = status
            if not os.path.exists(Config.wd_path):
                os.mkdir(Config.wd_path)
            data.to_csv(Config.wd_path + Config.inv_data_csv)
            return read_process_status
        except:
            err = "Failure reading data: " + str(sys.exc_info()[1])
            print(err)
            return err
        return "Data not loaded."

    def cgr(self, start, end, period):
        gr = round( (pow(end/start,(1/period)) - 1)*100, 2)
        return gr


class Reader():

    def read_files(self, file_list):
        bank_dict = {k: {'Count': 0, 'Files': []} for k in Config.banks}
        dfs = []
        cleaner = Cleaner()
        for f in file_list:
            for bank in Config.banks:
                for tp in Config.types:
                    for acc in Config.accounts:
                        prefix = tp + Config.sep + bank + Config.sep + acc
                        if f.startswith(prefix):
                            bank_dict[bank]['Count'] += 1
                            f_path_full = Config.data_path_prefix+f
                            cdf = pd.read_csv(f_path_full, header=0, index_col=False)
                            cdf[Config.bank_col] = bank
                            cdf[Config.acc_type] = tp
                            cdf[Config.acc_col] = str(acc)
                            cdf = cleaner.clean_naming_columns(cdf, Config.name_map)
                            cdf = cleaner.clean_date_columns(cdf, f, tp)
                            cdf = cleaner.clean_sign_transactions(cdf, bank)
                            dfs.append(cdf)
        lq_data = pd.concat(dfs, ignore_index=True, axis=0)


        for bank in bank_dict:
            print(bank + " : " + str(bank_dict[bank]['Count']) + " files read.")
        print("Fields:")
        print(", ".join(list(lq_data.columns)))
        print("Data shape: "+str(lq_data.shape))
        df = lq_data
        df[Config.date_col] = pd.to_datetime(df[Config.date_col], format=Config.d_date_format)
        df.sort_values([Config.date_col], inplace=True)
        fields_id_duplicates = [Config.trans_date, Config.post_date, Config.description, Config.amount, Config.balance, Config.bank_col, Config.acc_col]
        cleaner.clean_duplicates(df, fields_id_duplicates)
        print(self.null_checker(df))
        # all categories
        cats = df[df[Config.amount].notna()][[Config.description, Config.category]].drop_duplicates().sort_values(Config.category)
        num_cats_found = len(cats[Config.category].unique())
        print("Categories found: "+str(num_cats_found))
        known_cats = len(Config.expenses+Config.non_expenses)
        print("Known Categories: " + str(known_cats))
        cats.to_csv(Config.wd_path+Config.found_categories_csv)
        print(Config.line_string)

        # Category cleanup
        if num_cats_found > known_cats or cats.isna().any():
            status = cleaner.clean_categories(df)

        return df, status

    def make_data_usable(self):
        pass

    def null_checker(self, df):
        message = ""
        #Check for null dates
        nd = df[df[Config.date_col].isnull()].shape[0]
        if nd!=0:
            message+="Null dates\n"
        #Check for null amounts in CC
        nc = df[(df[Config.acc_type]=='CC') & (df[Config.amount].isnull())].shape[0]
        if nc!=0:
            message+="Null amounts in CC\n"
        #Check for null balances in S
        ns = df[(df[Config.acc_type]=='S') & (df[Config.balance].isnull())].shape[0]
        if ns!=0:
            message+="Null balances in Savings\n"

        return message


    def read_inv_files(self, file_list):
        per_type_count = {k: 0 for k in Config.inv_types}
        dfs = []
        cleaner = Cleaner()
        for f in file_list:
            for tp in Config.inv_types:
                for acc in Config.inv_accounts:
                    prefix = tp + Config.sep + acc
                    if prefix in f:
                        per_type_count[tp] += 1
                        f_path_full = Config.data_path_prefix + f
                        bdf = pd.read_csv(f_path_full, header=0, index_col=False)
                        bdf[Config.acc_type] = tp
                        bdf[Config.acc_col] = acc
                        # Consistent Naming
                        bdf = cleaner.clean_naming_columns(bdf, Config.inv_name_map)
                        dfs.append(bdf)
        inv = pd.concat(dfs, ignore_index=True, axis=0)
        inv = cleaner.clean_investment_data(inv)
        inv = cleaner.validate_inv_data(inv, Config.inv_interest_fields)
        inv.to_csv(Config.wd_path + Config.inv_final_data_file)
        return inv, 'Investment data loaded: Success'



class Cleaner():
    def clean_naming_columns(self, cdf, name_map):
        for field in name_map.keys():
            for p_name in name_map[field]:
                names = [c for c in cdf.columns]
                if p_name in names:
                    cdf.rename(columns={p_name: field}, inplace=True)
                    break
        return cdf


    def clean_date_columns(self, cdf, f, type):
        if type == 'CC':
            cdf = cdf[cdf[Config.amount].notna()]
            cdf[Config.date_col] = cdf[Config.trans_date]
        else:
            cdf = cdf[cdf[Config.balance].notna()]
            if Config.post_date in cdf.columns:
                cdf[Config.date_col] = cdf[Config.post_date]
            elif Config.trans_date in cdf.columns:
                cdf[Config.date_col] = cdf[Config.trans_date]
            else:
                print(f, "No Dates Found: " + str([x for x in cdf.columns]))
        return cdf


    def clean_sign_transactions(self, df, bank):
        # Positive Negative corrections
        # expenses are positive, payments are negative
        if Config.amount in df.columns and Config.banks_paid_negative[bank]:
            df[Config.amount] = -df[Config.amount]
        return df

    def clean_duplicates(self, df, fields):
        df = df[~df.duplicated(fields, keep=False)]
        return df

    def make_time_series(self, data):
        data[Config.date_col] = pd.to_datetime(data[Config.date_col])

    def prepare_dataframe(self, data, fields, date_col, mandatory):
        if not isinstance(data[Config.date_col].values[0], datetime):
            data[Config.date_col] = pd.to_datetime(data[Config.date_col])
        t = data[data[mandatory].notna()][fields].sort_values(date_col)
        t[Config.year_col] = t[date_col].dt.year
        t[Config.month_col] = t[date_col].dt.month
        t[Config.day_col] = t[date_col].dt.day
        t[Config.dow_col] = t[date_col].dt.strftime('%A')
        t[Config.ym_col] = t[date_col].dt.strftime("%Y-%m")
        return t

    def clean_investment_data(self, t):

        if Config.trans_date in t.columns:
            if not t[t[Config.trans_date].isna()].empty:
                t = t[t[Config.trans_date].notna()]
            if not t[t[Config.trans_date].str.contains('as')].empty:
                t = t[~t[Config.trans_date].str.contains('as')]
            t[Config.date_col] = pd.to_datetime(t[Config.trans_date])
            t[Config.year_col] = t[Config.date_col].dt.year
            t[Config.month_col] = t[Config.date_col].dt.month
            t[Config.day_col] = t[Config.date_col].dt.day
            t[Config.dow_col] = t[Config.date_col].dt.strftime('%A')
            t[Config.ym_col] = t[Config.date_col].dt.strftime("%Y-%m")
        else:
            print("No Dates Found: " + str([x for x in t.columns]))

        if Config.price in t.columns and Config.amount in t.columns:
            t[Config.price] = t[Config.price].astype(str).apply(lambda x: float(x.replace(',','').replace('(','').replace(')','').replace('$','').strip()))
            t[Config.amount] = t[Config.amount].astype(str).apply(lambda x: float(x.replace(',','').replace('(','').replace(')','').replace('$','').strip()))

        t[Config.amount] = t[Config.price] * t[Config.qty]

        mask = t[Config.category].str.contains('Dividend|Tax')
        t[mask] = t[mask].assign(Symbol=lambda x: x['Details'] if len(x[Config.symbol]) > 5 else x[Config.symbol])
        t[mask] = t[mask].assign(Amount=t[mask].apply(lambda x: float(str(x[Config.description]).replace('$','')), axis=1))

        return t

    def check_duration_match(self, filled, data):
        more_data_filled = filled[Config.date_col].max() >= data[Config.date_col].max() and filled[Config.date_col].min()  <= data[Config.date_col].min()
        return more_data_filled

    def find_difference(self, filled, data):
        remaining = data[data[Config.date_col]>filled[Config.date_col].max()]
        return remaining


    def read_prep_filled_category_dataframe(self, path, sep='\t'):
        df = pd.read_csv(path, sep=sep)
        if Config.date_col in df.columns:
            df[Config.date_col] = pd.to_datetime(df[Config.date_col])
        df = self.prepare_dataframe(df, list(df.columns), Config.date_col, Config.amount)
        return df

    def validate_inv_data(self, df, fields):
        df[fields].drop_duplicates(inplace=True)
        return df[fields]

    def validate_data(self, df, fields):
        not_present = []
        for f in fields:
            if f not in df.columns:
                not_present.append(f)

        if len(not_present)>0:
            og_data = pd.read_csv(Config.wd_path + Config.data_csv)
            og_data[Config.date_col] = pd.to_datetime(og_data[Config.date_col])
            join_on = list((Counter(fields) - Counter(not_present)).elements())
            temp = pd.merge(df, og_data, how='left', suffixes=['', '_'],
                            on=join_on)
        else:
            temp=df
        if Config.subcategory in temp.columns:
            fields.append(Config.subcategory)
        result = temp[fields]
        c = Cleaner()
        result = c.prepare_dataframe(result, fields, Config.date_col, Config.amount)

        return result

    def get_categories_filled(self, df, fields):
        filled_category_file = Config.wd_path+Config.filled_category_file
        message = ""
        if not exists(filled_category_file):
            message+="Category file not found.\n"
            fill_category_file = Config.wd_path + Config.fill_category_file
            df[fields].sort_values([Config.date_col, Config.category, Config.description]).reset_index().to_csv(fill_category_file)
            message+="Fill in the category expenses file in folder: "+fill_category_file+"\n"
        else:
            filled = pd.read_csv(filled_category_file, sep='\t')
            filled[Config.date_col] = pd.to_datetime(filled[Config.date_col])
            if self.check_duration_match(filled, df):
                message += "Category file found fully filled: Success.\n"
                df = self.read_prep_filled_category_dataframe(filled_category_file)
                df = self.validate_data(df, Config.exp_interest_fields)
                df.to_csv(Config.wd_path+Config.final_data_file)
            else:
                message += "Category file found partially filled.\n"
                rem_df = self.find_difference(filled, df)
                rem_df[fields].sort_values([Config.date_col, Config.category, Config.description]).reset_index().to_csv(Config.fill_category_file)
                message += "Fill the remaining unknown categories.\n"

        return message


    def check_period_bank_match(self, df):
        get_filled = False
        bank_uploaded = False
        if os.path.isfile(Config.wd_path+Config.filled_category_file):
            filled_category_file = Config.wd_path+Config.filled_category_file
            filled = pd.read_csv(filled_category_file, sep='\t')
            if filled[Config.date_col].max() >= df[Config.date_col].max().strftime('%Y-%m-%d'):
                for b in Config.banks:
                    filled_b_data = filled[filled[Config.bank_col]==b]
                    df_b_data = df[df[Config.bank_col] == b]
                    if filled_b_data[Config.date_col].max() >= df_b_data[Config.date_col].max().strftime('%Y-%m-%d'):
                        continue
                    else:
                        get_filled = True
                        bank_uploaded = True
                        break
            else:
                get_filled = True
            if get_filled:
                cols = Config.transaction_use_fields
                filled[cols].to_csv(Config.wd_path + Config.archive_filled_csv_file)
                os.remove(filled_category_file)
                rem = df[df[Config.date_col] > filled[Config.date_col].max()] if not bank_uploaded else df_b_data[df_b_data[Config.date_col] > filled_b_data[Config.date_col].max()]
                return rem
        else:
            return df


    def clean_categories(self, df):
        # Reduce the unknown categories
        self.make_time_series(df)
        tfields = Config.transactions_fields
        tf = self.prepare_dataframe(df, tfields, Config.date_col, Config.amount)
        sfields = Config.savings_fields
        sf = self.prepare_dataframe(df, sfields, Config.date_col, Config.balance)
        sf[Config.balance] = sf[Config.balance].astype('str')
        sf[Config.balance] = sf[Config.balance].str[:-1].astype('float64')

        rem = self.check_period_bank_match(tf)
        if rem is not None:
            rem[Config.subcategory] = pd.Series()
            tfields = Config.transaction_use_fields
            status = self.get_categories_filled(rem, tfields)
        else:
            status = self.get_categories_filled(tf, tfields)
        return status



class Plotter():
    def plot_income(self):
        print("Income")






class InvestorFunctions():

    def __init__(self, use_latest=False):
        self.use_latest = use_latest

    def find_price(self, fund):
        mkb_util = MarketBeatScraper()
        return mkb_util.get_price_symbol(fund)

    def get_fund_last_price(self, sym, is_stock):
        cur_price_file = Config.history_path + 'stck_current_price_list.csv' if is_stock else Config.history_path + 'indx_current_price_list.csv'
        if os.path.isfile(cur_price_file):
            prices = pd.read_csv(cur_price_file)
            latest = prices[prices[Config.symbol] == sym][Config.price].values[0]
            return latest
        return

    def get_fund_current_price(self, fund):
        try:
            if '-' in fund:
                fund = fund.split(' ')[0]
            ticker = yf.Ticker(fund)
            fund_df = ticker.history(period='2d')
            if fund_df.empty:
                last = self.find_price(fund)
            else:
                last = fund_df['Close'].tail(1).values[0]
            return last

        except:
            print("Fund data not found: " + fund + ' - ' + str(sys.exc_info()[1]))
            return None

    def analyze_investment_portfolio(self, bro, ret):
        stock_qtys = bro[bro[Config.qty].notna()].groupby(Config.symbol)[Config.qty].sum()
        brokerage = pd.DataFrame(bro[bro[Config.qty].notna()].groupby(Config.symbol)[Config.qty, Config.amount].sum())
        brokerage = brokerage[brokerage[Config.amount] != 0]
        brokerage[Config.amount] = round(brokerage[Config.amount], 2)
        brokerage_div = bro[bro[Config.category].str.contains('Dividend|Tax')].groupby(Config.symbol)[Config.amount].sum()
        for sym in brokerage.index.unique():
            try:
                if self.use_latest:
                    hist = self.get_fund_current_price(sym)
                else:
                    hist = self.get_fund_last_price(sym, True)
                if hist is not None:
                    brokerage.loc[sym, Config.current_amount] = round(stock_qtys.loc[sym] * hist, 2)
                    brokerage.loc[sym, Config.current_amount] += brokerage_div[sym] if sym in brokerage_div else 0
                    brokerage.loc[sym, Config.fund_gain] = round(brokerage.loc[sym, 'CurrentAmount'] - brokerage.loc[sym, Config.amount], 2)
                    brokerage.loc[sym, Config.fund_gain_perct] = round(brokerage.loc[sym, Config.fund_gain] * 100 / brokerage.loc[sym, Config.amount], 2)
                    brokerage.loc[sym, 'Cost_Basis'] = round(brokerage.loc[sym, Config.amount] / stock_qtys.loc[sym], 2)
                    brokerage.loc[sym, 'Current_Price'] = hist
            except:
                print(sym + " : " + str(sys.exc_info()[1]))
        brokerage[Config.percentage] = round(brokerage[Config.current_amount] * 100 / brokerage[Config.current_amount].sum(), 2)
        brokerage = brokerage.sort_values(Config.fund_gain_perct, ascending=False)

        my_spend_retirement = pd.DataFrame(ret[ret[Config.description].str.contains('Employee')].groupby(Config.symbol)[Config.amount].sum())
        my_spend_retirement.rename(columns={Config.amount: Config.my_spend}, inplace=True)
        retirement = ret[ret[Config.category].str.contains('Purchase')].groupby(Config.symbol)[Config.qty, Config.amount].sum()  # includes interest + div
        retirement_int = ret[(ret[Config.description].str.contains('Interest|Dividend'))].groupby(Config.symbol)[Config.amount].sum()
        retirement = pd.merge(retirement, my_spend_retirement, on=Config.symbol)
        retirement[Config.my_spend] = round(retirement[Config.my_spend], 2)
        retirement['Purchase_' + Config.percentage] = round(
            retirement[Config.amount] * 100 / retirement[Config.amount].sum(), 2)
        retirement['MyCost_' + Config.percentage] = round(
            retirement[Config.my_spend] * 100 / retirement[Config.my_spend].sum(), 2)
        indexes = list(retirement.index.unique())
        for sym in indexes:
            try:
                if self.use_latest:
                    hist = self.get_fund_current_price(sym)
                else:
                    hist = self.get_fund_last_price(sym, False)
                if hist is not None:
                    intr = retirement_int[sym] if sym in retirement_int else 0
                    retirement.loc[sym, 'CurrentAmount'] = round(retirement.loc[sym, Config.qty] * hist, 2) + intr
                    retirement.loc[sym, Config.my_fund_gain] = round(
                        retirement.loc[sym, 'CurrentAmount'] - retirement.loc[sym, Config.my_spend], 2)
                    retirement.loc[sym, Config.my_fund_gain_perct] = round(
                        retirement.loc[sym, Config.my_fund_gain] * 100 / retirement.loc[sym, Config.my_spend], 2)
                    retirement.loc[sym, 'Cost_Basis'] = round(
                        retirement.loc[sym, Config.amount] / retirement.loc[sym, Config.qty], 2)
                    retirement.loc[sym, Config.fund_gain] = round(
                        retirement.loc[sym, 'CurrentAmount'] - retirement.loc[sym, Config.amount], 2)
                    retirement.loc[sym, Config.fund_gain_perct] = round(
                        retirement.loc[sym, Config.fund_gain] * 100 / retirement.loc[sym, Config.amount], 2)
                    retirement.loc[sym, 'Current_Price'] = hist
            except:
                print(sym + " : " + str(sys.exc_info()[1]))
        retirement['Value_' + Config.percentage] = round(retirement[Config.current_amount] * 100 / retirement[Config.current_amount].sum(), 2)
        retirement = retirement.sort_values('Value_' + Config.percentage, ascending=False)


        return brokerage, retirement


    def sectype_analysis(self, brokerage, retirement):
        combined = pd.concat([brokerage, retirement], axis=0).reset_index()
        combined.loc[combined[Config.symbol].isin(Config.etf_list), Config.sec_type_col] = Config.etf
        combined.loc[combined[Config.symbol].str.contains('Idx|Index|401', na=False) & combined[
            Config.sec_type_col].isna(), Config.sec_type_col] = Config.index
        combined.loc[combined[Config.symbol].str.contains('Bond|BND|IUSB', na=False), Config.sec_type_col] = Config.bond
        combined.loc[combined[Config.symbol].str.contains('Real', na=False), Config.sec_type_col] = Config.real_estate
        combined.loc[combined[Config.sec_type_col].isna(), Config.sec_type_col] = Config.stock
        grouped = round(combined.groupby(Config.sec_type_col)[Config.current_amount].sum(), 2)

        stocks = list(combined[combined[Config.sec_type_col]==Config.stock][Config.symbol])
        etfs = list(combined[combined[Config.sec_type_col]==Config.etf][Config.symbol])
        index = list(combined[combined[Config.sec_type_col]==Config.index][Config.symbol])
        bond = list(combined[combined[Config.sec_type_col]==Config.bond][Config.symbol])
        real_est = list(combined[combined[Config.sec_type_col]==Config.real_estate][Config.symbol])
        elements = {Config.stock: [len(stocks), stocks], Config.index: [len(index), index], Config.etf: [len(etfs), etfs], Config.bond: [len(bond), bond], Config.real_estate: [len(real_est), real_est]}
        return elements, combined
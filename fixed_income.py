import pandas as pd
import dxlib as dx

future_list = ["DI1F26", "DI1F29"]
brazilian_dates = {"Jan": "Jan", "Fev": "Feb", "Mar": "Mar", "Abr": "Apr", "Mai": "May", "Jun": "Jun",
                   "Jul": "Jul", "Ago": "Aug", "Set": "Sep", "Out": "Oct", "Nov": "Nov", "Dez": "Dec"}

ipca_list = {"IPCA2029": "Tesouro_IPCA+_15-05-2029.csv",
             "IPCA2050": "Tesouro_IPCA+_com_Juros_Semestrais_15-08-2050.csv"}


def create_history(schema, security_name, product):
    security = dx.Security(security_name)
    product.index = pd.MultiIndex.from_product([product.index, [security]])
    schema.extend(dx.Schema(security_manager=dx.SecurityManager.from_list([security])))
    product.index.names = ['date', 'security']
    history = dx.History(product, schema)
    return history


class IPCA:
    def __init__(self):
        self.fixeds = {}

    def preprocess(self, schema):
        fixeds = {}

        for fixed_name, filename in ipca_list.items():
            fixed = pd.read_csv(filename)
            fixed.rename(columns={"Preco Medio": "close"}, inplace=True)
            fixed["Data Base"] = pd.to_datetime(fixed["Data Base"], format="%Y-%m-%d")
            fixed = fixed.set_index("Data Base")[['close']]
            fixeds[fixed_name] = create_history(schema, fixed_name, fixed)

        self.fixeds = fixeds
        return self

    def values(self):
        return self.fixeds.values()

    def __getitem__(self, item):
        return self.fixeds[item]


class Futures:
    def __init__(self):
        self.futures = {}

    def preprocess(self, schema):
        futures = {}

        for future_name in future_list:
            future = pd.read_csv(f"{future_name}.csv")
            future.rename(columns={"Fechamento": "close"}, inplace=True)
            future = future.replace({"Data": brazilian_dates}, regex=True)
            future["Data"] = pd.to_datetime(future["Data"], format="%d %b %Y")
            future = future.set_index("Data")[['close']]
            futures[future_name] = create_history(schema, future_name, future)

        ibov_name = "IND1!"
        ibov_filename = "Dados Históricos - Ibovespa Futuros.csv"
        cnybrl = "CNYBRL=X"
        cnybrl_filename = "CNY_BRL Dados Históricos.csv"
        futures[ibov_name] = self.preprocess_extra(schema, ibov_name, ibov_filename)
        futures[cnybrl] = self.preprocess_extra(schema, cnybrl, cnybrl_filename)

        self.futures = futures

        return self

    @staticmethod
    def preprocess_extra(schema, name, filename):
        future = pd.read_csv(filename, thousands=".", decimal=",", parse_dates=["Data"], dayfirst=True)

        future.rename(columns={"Último": "close"}, inplace=True)
        future = future.set_index("Data")[['close']]
        return create_history(schema, name, future)

    def __getitem__(self, item):
        return self.futures[item]

    def values(self):
        return self.futures.values()

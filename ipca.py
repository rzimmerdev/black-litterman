import pandas as pd
import matplotlib.pyplot as plt
import requests

if input("Download file? (y/n): ").lower() == "y":

    url = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059"
           "-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
    r = requests.get(url)
    with open("PrecoTaxaTesouroDireto.csv", "wb") as f:
        f.write(r.content)

df = pd.read_csv("PrecoTaxaTesouroDireto.csv", sep=";", decimal=",")

# print name of all securities that contains both 2029 and 2050 in Data Vencimento
print(f"Securities with 2029 and 2050")
print(df[df["Data Vencimento"].str.contains("2029") & df["Data Vencimento"].str.contains("2050")]["Tipo Titulo"].unique())

# print all securities
print(f"Available securities: {df['Tipo Titulo'].unique()}")
name = input("Enter the security name: ").replace("'", "").replace('"', "")
df = df[df["Tipo Titulo"] == name].copy()

maturities = df['Data Vencimento'].unique()
maturities.sort()
print(f"Available maturities: {maturities}")
maturity = input("Enter the maturity date (dd/mm/yyyy): ").replace("'", "").replace('"', "")
df = df[df["Data Vencimento"] == maturity].copy()

# make price avg of PU Venda Manha and PU Compra Manha
df["Preco Medio"] = (df["PU Venda Manha"] + df["PU Compra Manha"]) / 2
# format date column
df["Data Base"] = pd.to_datetime(df["Data Base"], format="%d/%m/%Y")
df.set_index("Data Base", inplace=True)
df.sort_index(inplace=True)

# use only values from last 12 months
df = df[df.index >= df.index[-1] - pd.DateOffset(years=1)]

plt.plot(df.index, df["Preco Medio"])
plt.show()

save = input("Save to csv? (y/n): ")
if save.lower() == "y":
    filename = f"{name}_{maturity}.csv".replace(" ", "_").replace("/", "-")
    df.to_csv(filename)
    # save plot
    plt.plot(df.index, df["Preco Medio"])
    plt.savefig(f"{filename}.png")
    print("File saved.")

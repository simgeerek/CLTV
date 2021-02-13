
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_

df.head()

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

df['TotalPrice'] = df['Quantity'] * df['Price']

df["Customer ID"].nunique()
df["Invoice"].nunique()
# 4339 müşteri toplam 18536 transaction yapmış

##############################################################
# Günlük İşlem Sayısı ve Grafiği
##############################################################

temp_df = df
temp_df['InvoiceDate'] = pd.to_datetime(temp_df['InvoiceDate']).dt.date
daily_transactions = temp_df.groupby("InvoiceDate").agg({"Invoice": lambda x: x.nunique()})\
                                                   .rename(columns={'Invoice':'Transactions'})

daily_transactions.head()

daily_transactions.index.min()
daily_transactions.index.max()
# Veri setindeki transactionları incelediğimizde, ilk işlemin 1 Aralık 2010'da gerçekleştiğini ve
# son işlemin 9 Aralık 2011'de gerçekleştiğini görüyoruz. 1 yıldan biraz daha uzun süreli bir veri kümesi diyebiliriz.


# GRAFIK
sns.set(rc={'figure.figsize':(11, 4)})
ax = daily_transactions["Transactions"].plot()
ax.set_ylabel('Daily Transaction Count')
plt.show()

##############################################################
# Aylık İşlem Sayısı ve Grafiği
##############################################################

temp_df['InvoiceDate'] = pd.to_datetime(temp_df['InvoiceDate']).dt.to_period('M')
monthly_transactions = temp_df.groupby("InvoiceDate").agg({"Invoice": lambda x: x.nunique()})\
                                                     .rename(columns={'Invoice':'Transactions'})

monthly_transactions.head()

# GRAFIK
ax = monthly_transactions["Transactions"].plot()
ax.set_ylabel('Monthly Transaction Count')
plt.show()
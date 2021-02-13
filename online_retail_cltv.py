
##############################################################
# CUSTOMER LIFETIME VALUE
##############################################################

# Her müşteri eşit değere sahip değildir. Bazı müşteriler diğerlerinden daha fazla gelir sağlar.
# Karı en üst düzeye çıkarmak istiyorsak hangisine odaklanıcağımızı ve yatırım yapacağımızı bilmemiz gerekir.

# Bir çok şirket müşterilerin diğerlerine kıyasla ne kadar değerli olduğunu belirlemek için CLTV adı verilen
# bir hesaplama kullanır.

# CLTV kişinin müşteri olarak kaldığı süre boyunca şirkete kazandıracağı parasal değerdir, kar miktarıdır.

# CLTV'yi müşteriler arasında karşılaştırarak, hangisinin sizin için daha fazla veya daha az karlı olduğunu belirleyebilir,
# böylece müşteri tabanınızı segmentlere ayırabilirsiniz.

# Her müşterinin karlılığını bilmek, onları yönetmenin ilk adımıdır. Ardından, pazarlama, ürün geliştirme, müşteri edinme
# ve elde tutma çabalarınızı nereye odaklayacağınıza karar verebilirsiniz.

# CLTV'yi hesaplamanın birden fazla yolu vardır.


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
# MUSTERI METRIKLERININ HESAPLANMASI
##############################################################

# Üzerinde çalıştığımız veri kümesi, ham işlem geçmişinden oluşur.
# Müşteri başına birkaç ölçüm türetmemiz gerekir:

# Frequency -> müşterinin ilk satın alma tarihinden sonra müşterinin satın aldığı tarihlerin sayısı
# Age(T) -> zaman birimi, bir müşterinin ilk satın alma tarihinden geçerli tarihe (veya veri kümesindeki son tarihe) kadar
# Recency -> müşterinin son satın almasından itibarenki yaşı

# Müşteri yaşı gibi ölçümleri hesaplarken, veri kümesinin sona erdiğini dikkate almamız gerektiğini unutmamak önemli.
# Bugünün tarihine göre hesaplamak hatalı sonuçlara sebep olabilir. Bunu göz önünde bulundurarak, veri kümesindeki
# son tarihi belirleyeceğiz ve bunu tüm hesaplamalar için bugünün tarihi olarak tanımlayacağız.

df.head()

observation_period_end = dt.datetime(2011, 12, 9)

summary_data_from_transaction_data = df.groupby('Customer ID').agg({'Invoice': lambda num: num.nunique(),
                                                                    'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                                    lambda date: (observation_period_end - date.min()).days],
                                                                    'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

summary_data_from_transaction_data["TotalPrice"] = summary_data_from_transaction_data["TotalPrice"] / summary_data_from_transaction_data["Invoice"]
summary_data_from_transaction_data.columns = summary_data_from_transaction_data.columns.droplevel(0)
summary_data_from_transaction_data.columns = ["frequency","recency","T","monetary_value"]

summary_data_from_transaction_data.head()


##############################################################
# Metriklerin Incelenmesi
##############################################################

# İlk alışveriş dahil.
# Monetary_value değeri her bir faturanın TotalPrice'larının toplamının fatura sayısına bölümünden hesaplanıyor.

# 12347 ID'li müşteri için sağlamasını yapalım
customer = df[df["Customer ID"] == 12347.0].groupby("Invoice").agg({"TotalPrice":"sum","InvoiceDate":"max"})
customer.shape[0] # beklenen frekans
customer["TotalPrice"].sum() / customer.shape[0] # beklenen monetary_avg

# 12349.0 ID'li müşteri için sağlamasını yapalım
customer = df[df["Customer ID"] == 12349.0].groupby("Invoice").agg({"TotalPrice":"sum","InvoiceDate":"max"})
customer.shape[0] # beklenen frekans
customer["TotalPrice"].sum() / customer.shape[0] # beklenen monetary_avg


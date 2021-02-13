
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
daily_transactions = temp_df.groupby("InvoiceDate").agg({"Invoice": lambda x: x.nunique()}).rename(columns={'Invoice':'Transactions'})
daily_transactions.head()
daily_transactions.index.min()
daily_transactions.index.max()
# Veri setindeki transactionları incelediğimizde, ilk işlemin 1 Aralık 2010'da gerçekleştiğini ve
# son işlemin 9 Aralık 2011'de gerçekleştiğini görüyoruz. 1 yıldan biraz daha uzun süreli bir veri kümesi diyebiliriz.

# GRAFİK
sns.set(rc={'figure.figsize':(11, 4)})
ax = daily_transactions["Transactions"].plot()
ax.set_ylabel('Daily Transaction Count')
plt.show()

##############################################################
# Aylık İşlem Sayısı ve Grafiği
##############################################################

temp_df['InvoiceDate'] = pd.to_datetime(temp_df['InvoiceDate']).dt.to_period('M')
monthly_transactions = temp_df.groupby("InvoiceDate").agg({"Invoice": lambda x: x.nunique()}).rename(columns={'Invoice':'Transactions'})
monthly_transactions.head()

# GRAFİK
ax = monthly_transactions["Transactions"].plot()
ax.set_ylabel('Monthly Transaction Count')
plt.show()


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



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
import numpy as np
import datetime as dt
from sklearn.metrics import mean_squared_error
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from lifetimes.plotting import plot_period_transactions
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df.head()

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

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


##############################################################
# HOLDOUT
##############################################################

# Veri setini train ve test olarak ayırmamız gerekiyor.
# 12 aylık bir veri seti
# Train -> ilk 8 ay
# Test -> son 4 ay

# TRAIN
calibration_period_end = dt.datetime(2011, 8, 8)  # Train seti için sınır
observation_period_end = dt.datetime(2011, 12, 9)# Test seti için sınır


calibration_transactions = df.loc[df["InvoiceDate"] <= calibration_period_end]

calibration_summary_data = calibration_transactions.groupby('Customer ID').agg({'Invoice': lambda num: num.nunique(),
                                                                                'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                                                lambda date: (calibration_period_end - date.min()).days],
                                                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

calibration_summary_data["TotalPrice"] = calibration_summary_data["TotalPrice"] / calibration_summary_data["Invoice"]
calibration_summary_data.columns = calibration_summary_data.columns.droplevel(0)
calibration_summary_data.columns = ["frequency_cal","recency_cal","T_cal","monetary_value_cal"]

calibration_summary_data.head()

# TEST
holdout_transactions = df.loc[(observation_period_end >= df["InvoiceDate"]) & (df["InvoiceDate"] > calibration_period_end)]

holdout_summary_data = holdout_transactions.groupby('Customer ID').agg({'Invoice': lambda num: num.nunique(),
                                                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum(),
                                                                        'InvoiceDate': lambda date: (observation_period_end - calibration_period_end).days})

holdout_summary_data["TotalPrice"] = holdout_summary_data["TotalPrice"] / holdout_summary_data["Invoice"]
holdout_summary_data.columns = ["frequency_holdout","monetary_value_holdout","duration_holdout"]

holdout_summary_data.head()

# COMBINED
combined_data = calibration_summary_data.join(holdout_summary_data, how="left")
combined_data.fillna(0, inplace=True)

combined_data.head()

combined_data = combined_data.loc[combined_data.frequency_cal > 1, :]
#combined_data = combined_data.loc[combined_data.monetary_value_cal> 0, :]

combined_data.head()

#combined_data.isnull().any()


##############################################################
# BG-NBD MODEL
##############################################################
# Model, müşteriler için beklenen tekrar ziyaretlerini hesaplamak için kullanılır.
# Ayrıca, bir müşterinin churn olup olmayacağını belirlemek için de kullanılabilir.


#Train the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(combined_data['frequency_cal'], combined_data['recency_cal'], combined_data['T_cal'])

#Predict
predicted_freq = bgf.predict(combined_data['duration_holdout'], # Tahmin için gün sayısı
                        combined_data['frequency_cal'],
                        combined_data['recency_cal'],
                        combined_data['T_cal'])

# Actual values ile predicted values gözlemlemek için yeni bir dataframe oluşturma
df_comp_freq = pd.DataFrame()
df_comp_freq ["ActualFrequency"] = combined_data['frequency_holdout']
df_comp_freq ["Predicted"] = predicted_freq
df_comp_freq.head(20)

# Elimizdeki gerçek ve tahmin edilen değerlerle bazı standart değerlendirme ölçütlerini hesaplayabiliriz.
def score_model(actuals, predicted, metric='mse'):
    # make sure metric name is lower case
    metric = metric.lower()
    # Mean Squared Error and Root Mean Squared Error
    if metric == 'mse' or metric == 'rmse':
        val = np.sum(np.square(actuals - predicted)) / actuals.shape[0]
        if metric == 'rmse':
            val = np.sqrt(val)
    # Mean Absolute Error
    elif metric == 'mae':
         val = np.sum(np.abs(actuals - predicted)) / actuals.shape[0]
    else:
        val = None
    return val

# score the model
print('MSE: {0}'.format(score_model(combined_data["frequency_holdout"], predicted_freq, 'mse')))

# Modelleri karşılaştırmak için önemli olsa da, MSE metriğini herhangi bir modelin genel fit iyiliği açısından yorumlamak biraz daha zordur.
# Modelimizin verilerimize ne kadar iyi fit olduğuna dair daha fazla bilgi sağlamak için, bazı gerçek ve tahmin edilen değerler arasındaki ilişkileri görselleştirelim.
plot_calibration_purchases_vs_holdout_purchases(bgf, combined_data)
plt.show()

plot_period_transactions(bgf)
plt.show()

# Tahmin edilen frekans değeri combined_data'ya ekleme
combined_data["frequency_predict"] = predicted_freq
combined_data.head()


##############################################################
# GAMMA GAMMA MODEL
##############################################################

# Gamma Gamma'yı kullanabileceğimizden emin olmak için, frekans ve parasal değerlerin
# ilişkili olup olmadığını kontrol etmemiz gerekir. (?)
combined_data[['monetary_value_cal', 'frequency_cal']].corr()
# Korelasyon düşük, devam

#Model fit
ggf = GammaGammaFitter(penalizer_coef = 0.01)
ggf.fit(combined_data['frequency_cal'],
        combined_data['monetary_value_cal'])

#Prediction
monetary_pred = ggf.conditional_expected_average_profit(combined_data['frequency_holdout'],
                                                        combined_data['monetary_value_holdout'])

# Actual values ile predicted values gözlemlemek için yeni bir dataframe oluşturma
df_comp_m = pd.DataFrame()
df_comp_m["ActualMonetary"] = combined_data['monetary_value_holdout']
df_comp_m["Predicted"] = monetary_pred
df_comp_m.head(20)

print("Expected Average Sales: %s" % monetary_pred.mean())
print("Actual Average Sales: %s" % combined_data["monetary_value_holdout"].mean())
print("Difference: %s" % (combined_data["monetary_value_holdout"].mean() - monetary_pred.mean()))
print("Mean Squared Error: %s" % mean_squared_error(combined_data["monetary_value_holdout"],monetary_pred))
print("Root Mean Squared Error: %s" % np.sqrt(mean_squared_error(combined_data["monetary_value_holdout"],monetary_pred)))

# Actual ve predicted monetary avg. grafiği
plt.figure(figsize=(10, 7))
plt.scatter(monetary_pred, combined_data['monetary_value_holdout'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual')
plt.show()

# Tahmin edilen monetary değeri combined_data'ya ekleme
combined_data["monetary_value_predict"] = monetary_pred
combined_data.head()


##############################################################
# CLV MODEL
##############################################################
# Bu model expected purchase tahmini alacak ve expected purchase value ile birleştirecektir.
# Belirli bir süre içinde bir müşterinin  ne kadar değerli olduğuna dair bir tahmine ulaşılmasını sağlar.

clv = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    combined_data['frequency_cal'],
    combined_data['recency_cal'],
    combined_data['T_cal'],
    combined_data['monetary_value_cal'],
    time=4, # months
    freq="D", # T'nin frekans bilgisi
    discount_rate=0.01
)

clv.head()
combined_data["CLV"] = clv
combined_data.head(20)

# Bunlar ilk 10 en değerli müşteri önümüzdeki 4 ay için
combined_data.sort_values('CLV', ascending=False).head(10)

# CLV modelinin performansını nasıl değerlendiririz?
# Simple bir baseline ile karşılaştırabiliriz.
# Target olarak en iyi müşterilerin %20 sini seçelim.
# En çok satın alım gerçekleştiren müşterileri seçmek önemli.

combined_data.head(20)
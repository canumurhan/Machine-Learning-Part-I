#Regresyon Modelleri için Hata Değerlendirme
#Çalışanların deneyim yılı ve maaş bilgileri verilmiştir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: "%.1f" %x)

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_excel("datasets/maas_tahmin.xlsx")

df.shape
df.head()
df.isnull().sum()
df.describe().T

#ADIM 1:Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
#Bias=275,Weight=90(y’=b+wx)
X=df["deneyim_yili(x)"]
y_=275 + 90 * X


#ADIM2:Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş
#tahmini yapınız.

y_heat = []
for x in df["deneyim_yili(x)"]:
    y_heat.append(275 + 90 * x)


df["y_heat"] = y_heat

df.head()

#Modelin başarısını ölçmek için MSE,RMSE,MAE skorlarını hesaplayınız.

#MSE
mean_squared_error(df["maas(y)"],y_heat)

df["maas(y)"].mean()
df["maas(y)"].std()

#MAE

#MAE
mean_absolute_error(df["maas(y)"],y_heat)

#RMSE

np.sqrt(mean_squared_error(df["maas(y)"],y_heat))






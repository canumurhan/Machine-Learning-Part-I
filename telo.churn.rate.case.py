#TelcoChurnPrediction
#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
#geliştirilmesi beklenmektedir.

#Telco müşterikaybıverileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu
#ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
#Hangi müşterilerin hizmetlerinden ayrıldığını,kaldığını veya hizmete kaydolduğunu gösterir

#CustomerId:Müşteri İd’si
#Gender:Cinsiyet
#SeniorCitizen:Müşterinin yaşlı olup olmadığı(1, 0)
#Partner:Müşterinin bir ortağı olup olmadığı(Evet, Hayır)
#Dependents:Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı(Evet, Hayır)
#tenure:Müşterininşirkettekaldığıay sayısı
#PhoneService:Müşterinintelefonhizmetiolupolmadığı(Evet, Hayır)
#MultipleLines:Müşterininbirdenfazlahattıolupolmadığı(Evet, Hayır, Telefonhizmetiyok)
#InternetServiceMüşterinininternet servissağlayıcısı(DSL, Fiber optik, Hayır)
#OnlineSecurityMüşterininçevrimiçigüvenliğininolupolmadığı(Evet, Hayır, İnternet hizmetiyok)
#OnlineBackupMüşterininonline yedeğininolupolmadığı(Evet, Hayır, İnternet hizmetiyok)
#DeviceProtection:Müşterinincihazkorumasınasahipolupolmadığı(Evet, Hayır, İnternet hizmetiyok)
#TechSupport:Müşterininteknikdestekalıpalmadığı(Evet, Hayır, İnternet hizmetiyok)
#StreamingTV:MüşterininTV yayınıolupolmadığı(Evet, Hayır, İnternet hizmetiyok)
#StreamingMovies:Müşterininfilm akışıolupolmadığı(Evet, Hayır, İnternet hizmetiyok)
#Contract:Müşterininsözleşmesüresi(Aydan aya, Bir yıl, İkiyıl)
#PaperlessBilling:Müşterininkağıtsızfaturasıolupolmadığı(Evet, Hayır)
#PaymentMethod:Müşterininödemeyöntemi(Elektronikçek, Posta çeki, Banka havalesi(otomatik)Kredikartı(otomatik))
#MonthlyCharges:Müşteridenaylıkolaraktahsil edilentutar
#TotalCharge:Müşteridentahsil edilentoplamtutar
#Churn:Müşterininkullanıpkullanmadığı(Evet veyaHayır)

#Görev 1 : Keşifçi Veri Analizi
#Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: "%.1f" %x)

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


df=pd.read_csv("datasets/Telco-Customer-Churn.csv")
df.head()
df.describe().T
df.isnull().sum()
df.info()

###################
# grab_col_names
###################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


num_cols = [col for col in num_cols if col not in "customerID"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
mask = df["TotalCharges"].str.strip() == ""
rows_with_spaces = df[mask]
df.loc[mask, "TotalCharges"] = np.nan


df["TotalCharges"]=df["TotalCharges"].astype(float)
df["TotalCharges"].replace(" ", np.nan, inplace=True)

#Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
num_scatter_percentage = len(num_cols) / len(df.columns) * 100
cat_scatter_percentage = len(cat_cols) / len(df.columns) * 100
car_scatter_percentage = len(cat_but_car) / len(df.columns) * 100

output = f"""
Numerik Değişkenler: {num_scatter_percentage:.2f}%
Kategorik Değişkenler: {cat_scatter_percentage:.2f}%
Kardinal Değişkenler: {car_scatter_percentage:.2f}%
"""

print(output)

#Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
for col in cat_cols:
    print(df.groupby([col, 'Churn']).count())

#Adım 5: Aykırı gözlem var mı inceleyiniz.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.85):
    """
    Bir dataframe için verilen ilgili kolondaki aykırı değerleri tespit edebilmek adına üst ve alt limitleri belirlemeyi
    sağlayan fonksiyondur

    Parameters
    ----------
    dataframe: "Dataframe"i ifade eder.
    col_name: Değişkeni ifade eder.
    q1: Veri setinde yer alan birinci çeyreği ifade eder.
    q3: Veri setinde yer alan üçüncü çeyreği ifade eder.

    Returns
    -------
    low_limit, ve up_limit değerlerini return eder
    Notes
    -------
    low, up = outlier_tresholds(df, col_name) şeklinde kullanılır.
    q1 ve q3 ifadeleri yoru açıktır. Aykırı değerle 0.01 ve 0.99 değerleriyle de tespit edilebilir.

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Bir dataframein verilen değişkininde aykırı gözlerimerin bulunup bulunmadığını tespit etmeye yardımcı olan
    fonksiyondur.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(f"{col} : {check_outlier(df, col)}")
#Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()

#Görev 2 : Feature Engineering
#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df.dropna(inplace=True)

#Adım 2: Yeni değişkenler oluşturunuz.

# 1. Toplam Hizmet Sayısı
services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['TotalServices'] = df[services].apply(lambda x: (x != 'No') & (x != 'No internet service') & (x != 'No phone service')).sum(axis=1)

# 2. Sözleşme Süresi
contract_duration = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractDuration'] = df['Contract'].map(contract_duration)

# 3. Elektronik Ödeme
df['ElectronicPayment'] = df['PaymentMethod'].apply(lambda x: 1 if 'electronic' in x.lower() else 0)

# 4. Müşteri Sadakati
df['LoyaltyScore'] = df['tenure'].apply(lambda x: x // 12)  # Her 12 ay için 1 puan

# 5. Aylık ve Toplam Harcama Oranı
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SpendingRatio'] = df['MonthlyCharges'] / df['TotalCharges']
df['SpendingRatio'] = df['SpendingRatio'].replace([np.inf, -np.inf], np.nan)  # Sıfıra bölme hatalarını NaN ile değiştir

# 6. Yaşlı Müşteri ve Teknoloji Kullanımı
df['SeniorTechUse'] = ((df['SeniorCitizen'] == 1) &
                             (df['InternetService'] != 'No')).astype(int)

df.head()

#Adım 3: Encoding işlemlerini gerçekleştiriniz.

mapping = {"Yes": 1,
           "No": 0}

yes_no_cols = [col for col in df.columns if df[col].isin(["Yes", "No"]).all()]

for col in yes_no_cols:
    df[col] = df[col].map(mapping)

other_cols = [col for col in df.columns if
              not (df[col].isin([0, 1]).all()) and (col != "customerID") and (
                          df[col].dtype not in ["float64", "int64"])]

df = pd.get_dummies(df, columns=other_cols, drop_first=True)
df = df.drop("customerID", axis=1)
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
y = df["Churn"]
X = df.drop("Churn", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)



#Görev 3 : Modelleme
#Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip.
#En iyi 4 modeli seçiniz.

knn_model = KNeighborsClassifier()
knn_model.fit(X, y)
knn_model.get_params()

y_pred = knn_model.predict(X)
y_prop = knn_model.predict_proba(X)

print(classification_report(y, y_pred))

roc_auc_score(y, y_pred)

cv_results = cross_validate(knn_model, X, y, cv=4, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()

cv_results["test_f1"].mean()

cv_results["test_roc_auc"].mean()

#Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve
#bulduğunuz hiparparametreler ile modeli tekrar kurunuz.

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,#işlemcileri tam performans çalıştırıyor.
                           verbose=1).fit(X, y) #verbose da rapor bekleyip beklemediğimizi sorar.1 yaparsak rapor verir.

knn_gs_best.best_params_

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)#Burada best parametreyi ** kullanarak atayabiliriz. Bu bize birden fazla best parametre yazmamız gerektiğinde kolaylık sağlayacaktır.

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=48,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()














import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from outlier_detection  import outliers_detect
from feature_engineering import drop_columns,create_feature,detect_vif



def run_eda(df):

    df_final = df.copy()
    print(df_final.head())
    print(df_final.tail())

    print(df_final.shape)
    # (7043 satır, 21 sütundan oluşmaktadır)
    print(df_final.dtypes)
    print(df_final.describe().T)

    df_final["Churn"] = df_final["Churn"].map({"Yes": 1, "No": 0})
    df_final["TotalCharges"] = pd.to_numeric(df_final['TotalCharges'], errors='coerce')
    print(df_final.isnull().sum())
    print((df_final == " ").sum())
    print(df_final.columns)

    cat_columns = ['SeniorCitizen', 'gender', 'Partner', 'Dependents',
                   'PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                   'PaymentMethod']
    for col in cat_columns:
        print(df_final.groupby(col)["Churn"].agg(["mean", "count"]))
        print("-----------")
        sns.barplot(x=col, y="Churn", data=df_final)
        plt.show()
        table = pd.crosstab(df_final[col], df_final["Churn"])
        chi2, p, dof, expected = chi2_contingency(table)
        print("p:", p)

        print("-----")
        print(df_final.loc[df_final["TotalCharges"].isnull(), ["tenure", "TotalCharges"]])

        df_final["TotalCharges"] = df_final["TotalCharges"].fillna(0)

        num_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_columns:
        print(col)
        print(df_final.groupby("Churn")[col].agg(["mean", "count"]))
        print("-------")
        sns.boxplot(x="Churn", y=col, data=df_final)
        plt.show()
        group1 = df_final[df_final["Churn"] == 0][col]
        group2 = df_final[df_final["Churn"] == 1][col]
        stat, p = ttest_ind(group1, group2, equal_var=False)
        print(col, "p:", p)
        print(col, "t:", stat)

    corr_matrix = df_final[num_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()

    x = df_final[["TotalCharges", "tenure"]]
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["vif"] = [variance_inflation_factor(x.values, i)
                       for i in range(len(x.columns))]
    print(vif_data)

    print(df_final.isnull().sum())

    outliers_detect(df_final, num_columns)
    df_final = drop_columns(df_final)
    df_final = create_feature(df_final)
    print(df_final.shape)
    print(df_final.head())
    print(df_final.columns)
    print(df_final.shape)
    detect_vif(df_final)
    return df_final











#SeniorCitizien değişkeni Churn ile anlamlı ilişki göstermektedir.(p<0.005)
#Senior olan müşterilerin Churn oranı istatiksel olarak daha yüksektir.

#Tenure değişkeni,Churn ile anlamlı ilişki göstermektedir.
#Tenure ile Churn arasında negatif yönlü ilişki bulunmaktadır.

#Monthly Charges,Churn ile anlamlı ilişki göstermektedir.
#Yüksek aylık ödeyen müşterilerin Churn etme oranı daha yüksektir.

#Phone service ve gender değişkenlerinin Churn ile anlamlı ilişkileri bulunmamaktadır.
#Contrat değişkeni ile Churn değişkeni arasında anlamlı bir ilişki bulunmaktadır.
#Month-to-month kontrata sahip müşterilerin,one year ve two year kontrata sahip müşterilere göre churn etme oranı daha yüksek.
#Electronic check ödeme yöntemini kullanan müşterilerin churn oranı daha yüksektir.
#Fiber optic InternetServicesi kullanan müşterilerin Churn oranı daha yüksektir.
#VIF değerleri 5 in üzerinde olduğu için TotalCharges ve Tenure arasında orta-yüksek korelasyon vardır.LOgistic Regresyon modeli kullanılırsa katsayılar kararsız olacağından TotalCharges değişkeni çıkarılmalıdır.
#totalcharges değeri nan olan değerler tespit edilmiştir. Eksik veriler tenure=0 olan veriler olduğu görülmüştür. Bu da henüz ödeme yapmamalarından kaynaklıdır. Eksik değerler 0 olarak güncellenmiştir.
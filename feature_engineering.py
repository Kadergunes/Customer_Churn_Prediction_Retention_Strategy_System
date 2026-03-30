import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def drop_columns(df_final):
    df_final=df_final.drop(["customerID","gender","PhoneService"],axis=1)
    return df_final


def create_feature(df_final):
    df_final=df_final.copy()
    #df_final["EntertainmentServices"]=((df_final["StreamingTV"]=="Yes").astype(int)+
    #                               (df_final["StreamingMovies"]=="Yes").astype(int))

    df_final["TotalServices"]=((df_final["MultipleLines"]=="Yes").astype(int)+
                               (df_final["OnlineSecurity"]=="Yes").astype(int)+
                               (df_final["OnlineBackup"]=="Yes").astype(int)+
                               (df_final["DeviceProtection"]=="Yes").astype(int)+
                               (df_final["TechSupport"]=="Yes").astype(int)+
                               (df_final["StreamingTV"]=="Yes").astype(int)+
                               (df_final["StreamingMovies"]=="Yes").astype(int))

    df_final["SecurityServices"]=((df_final["OnlineSecurity"]=="Yes").astype(int)+
                                  (df_final["OnlineBackup"]=="Yes").astype(int)+
                                  (df_final["DeviceProtection"]=="Yes").astype(int)+
                                  (df_final["TechSupport"]=="Yes").astype(int))

    #df_final.loc[df_final["tenure"] == 0, "AvgMonthlyCharge"] = 0
    #df_final["AvgMonthlyCharge"]=df_final["AvgMonthlyCharge"].fillna(df_final["TotalCharges"]/df_final["tenure"])

    #bins=[0,12,24,36,48,df_final["tenure"].max()]
    #labels=["0-12","12-24","24-36","36-48","49+"]
    #df_final["tenureGroup"]=pd.cut(df_final["tenure"],bins=bins,labels=labels,right=True)

    return df_final


def detect_vif(df_final):
    x = df_final[["TotalServices","tenure","MonthlyCharges"]]
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["vif"] = [variance_inflation_factor(x.values, i)
                       for i in range(len(x.columns))]
    print(vif_data)
    return vif_data





#def detect_vif_2(df_final):
#    x = df_final[baseline_feature]
#    vif_data = pd.DataFrame()
#    vif_data["feature"] = x.columns
#    vif_data["vif"] = [variance_inflation_factor(x.values, i)
#                       for i in range(len(x.columns))]
#    print(vif_data)


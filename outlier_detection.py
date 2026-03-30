import numpy as np

def outliers_detect(df_final,num_columns):
    print("outliers length:")
    for col in num_columns:
        Q1=df_final[col].quantile(0.25)
        Q3=df_final[col].quantile(0.75)

        IQR=Q3-Q1

        lower=Q1-1.5*IQR
        upper=Q3+1.5*IQR
        outliers=df_final[(df_final[col]<lower)| (df_final[col]>upper)]

        print(col,len(outliers))


#Outlier detection was performed using the IQR method on numerical features (tenure, MonthlyCharges, TotalCharges).
# No outliers were detected within the calculated IQR bounds.
# Therefore, no further anomaly detection methods such as Isolation Forest were applied.
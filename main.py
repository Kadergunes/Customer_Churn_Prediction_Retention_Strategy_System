import pandas as pd
from eda import run_eda
from modeling import run_modeling
from outlier_detection import outliers_detect
from evaluation import evaluate_model
from shap_analysis import run_shap
from config import advanced_model
from segmentation import get_segment,get_action


df = pd.read_csv('data.csv')

df_final=run_eda(df)


results=run_modeling(df_final)

baseline_pipeline,X_test_base,y_test_base,adv_pipeline,X_test_adv,y_test_adv=run_modeling(df_final)
#print(results)

evaluate_model(baseline_pipeline,X_test_base,y_test_base)
evaluate_model(adv_pipeline,X_test_adv,y_test_adv)
run_shap(adv_pipeline,df_final[advanced_model])
#print(sorted(df_final["MonthlyCharges"]))
#df_final=create_segmentation(df_final)

#print(df_final)
#print(analyze_segmentation(df_final))
#save_segmentation(df_final,filename="aksiyon önerisi")

#
import pandas as pd
import shap

def run_shap(pipeline,X):
    rf_model = pipeline.named_steps["model"]
    preprocessor=pipeline.named_steps["preprocessing"]
    X_encoded=preprocessor.transform(X)
    features_names=preprocessor.get_feature_names_out()



    explainer=shap.TreeExplainer(rf_model)
    shap_values=explainer.shap_values(X_encoded)
    shap.summary_plot(shap_values,X_encoded,feature_names=features_names)

# en etkili 3 feature:
# cat__Contract_Two year olan müşterilerin churn etme olasılığı düşüktür.
#num__tenure(müşteri olma süresü) fazla olan müşteriler daha sadıktır.
#num__MonthlyCharges (aylık fatura miktarı yükseldikçe müşterilerin ayrılma oalsılığı artmaktadır.)





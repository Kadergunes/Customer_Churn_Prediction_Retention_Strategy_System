About the Project
This project aims to predict customer churn using machine learning and transform these predictions into actionable business insights. 
The dataset was prepared through data preprocessing and feature engineering steps before modeling.

Modeling
A churn prediction model was developed using XGBoost and Scikit-learn, achieving approximately 0.85 ROC-AUC.
The model generates churn probabilities for each customer, enabling a more detailed analysis of risk levels.

Segmentation
Model outputs were combined with business rules to create a rule-based segmentation system. 
Customers are categorized into high, medium, and low risk segments based on variables such as
tenure, contract type, monthly charges, and churn probability.

Retention Strategy
Different actions were defined for each customer segment:
High Risk: Discount offers and contract renewal campaigns
Medium Risk: Loyalty programs and incentives
Low Risk:Customer satisfaction-focused actions

Model Interpretability
To ensure interpretability, SHAP was used to identify the key features influencing churn predictions.
<img width="804" height="632" alt="Ekran Resmi 2026-05-05 19 35 02" src="https://github.com/user-attachments/assets/7ac8bd5b-3c56-4d4a-b011-8851a932582c" />

Interface
An interactive user interface was developed using Gradio. Through this interface, users can:
Get churn predictions
View customer risk segments
Access recommended retention actions
<img width="1434" height="689" alt="Ekran Resmi 2026-05-05 19 20 55" src="https://github.com/user-attachments/assets/b04b49ff-1cd7-4cea-bc97-0e71f91211f7" />
<img width="1414" height="700" alt="Ekran Resmi 2026-05-05 19 21 15" src="https://github.com/user-attachments/assets/e219041c-58cc-4688-920e-63148e4f8f3a" />


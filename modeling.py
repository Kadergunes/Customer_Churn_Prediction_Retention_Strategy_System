import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import baseline_feature,advanced_model


def build_pipeline(X):
    cat_cols=X.select_dtypes(include="object").columns
    num_cols=X.select_dtypes(exclude="object").columns

    preprocessor=ColumnTransformer(
        transformers=[
            ("num",StandardScaler(),num_cols),
            ("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),cat_cols)
        ]
    )
    return preprocessor

def run_modeling(df):
    #print("run modeling başlatıldı.")
    y = df["Churn"]
    X_base = df[baseline_feature]
    preprocessor_base = build_pipeline(X_base)

    baseline_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor_base),
        ("model", LogisticRegression(max_iter=1000))
    ])

    X_adv = df[advanced_model]
    preprocessor_adv = build_pipeline(X_adv)

    adv_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor_adv),
        ("model", XGBClassifier(n_estimators=300,
                                learning_rate=0.05,
                                max_depth=6,
                                random_state=42))
    ])

    baseline_cv = cross_val_score(
        baseline_pipeline, X_base, y,
        cv=5, scoring="accuracy"
    )
    print("Baseline CV Mean:", baseline_cv.mean())
    print("Baseline CV Std:", baseline_cv.std())

    adv_cv = cross_val_score(
        adv_pipeline, X_adv, y,
        cv=5, scoring="accuracy"
    )
    print("Advanced CV Mean", adv_cv.mean())
    print("Adv CV Std:", adv_cv.std())

    X_train_base, X_test_base, y_train, y_test_base = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )
    baseline_pipeline.fit(X_train_base, y_train)

    X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
        X_adv, y, test_size=0.2, random_state=42
    )
    adv_pipeline.fit(X_train_adv, y_train_adv)


    return(
        baseline_pipeline,
        X_test_base,
        y_test_base,
        adv_pipeline,
        X_test_adv,
        y_test_adv


    )



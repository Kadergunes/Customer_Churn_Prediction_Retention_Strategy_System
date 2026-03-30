from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

def evaluate_model(pipeline,X_test,y_test):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred=pipeline.predict(X_test)
    y_pred=(y_prob>0.3).astype(int)


    print("Clasification Report:")
    print(classification_report(y_test,y_pred))

    print("\n Confusion Report:")
    print(confusion_matrix(y_test,y_pred))

    print("\n Roc-Auc Score:")
    print(roc_auc_score(y_test,y_prob))


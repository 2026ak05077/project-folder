import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.title("Adult Income Classification")

uploaded_file = st.file_uploader(
"Upload Test CSV",
    type = ['csv']
)

option = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.iloc[:,:-1]

    y = df.iloc[:,-1]

    model_paths = {
        "Logistic Regression":
        "model/logistic_regression.pkl",
    
        "Decision Tree":
        "model/decision_tree.pkl",
    
        "KNN":
        "model/knn.pkl",
    
        "Naive Bayes":
        "model/naive_bayes.pkl",
    
        "SVM":
        "model/svm.pkl"
    }

    model = joblib.load(
        model_paths[option]
    )
    
    pred = model.predict(X)
    
    proba = model.predict_proba(X)[:,1]
    
    st.write(
        "Accuracy",
        round(
            accuracy_score(y,pred),4
        )
    )
    
    st.write(
        "AUC",
        round(
            roc_auc_score(y,proba),4
        )
    )
    
    st.write(
        "Precision",
        round(
            precision_score(y,pred),4
        )
    )
    
    st.write(
        "Recall",
        round(
            recall_score(y,pred),4
        )
    )
    
    st.write(
        "F1",
        round(
            f1_score(y,pred),4
        )
    )
    
    st.write(
        "MCC",
        round(
            matthews_corrcoef(y,pred),4
        )
    )
    
    st.subheader("Confusion Matrix")
    
    st.write(confusion_matrix(y,pred))
    
    st.subheader("Classification Report")
    
    st.text(classification_report(y,pred))

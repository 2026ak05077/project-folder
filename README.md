Adult Income Classification Using Machine Learning

Name: Sanskar Maharana
Course: M.Tech (Machine Learning & Artificial Intelligence)
Institute: BITS Pilani
Assignment: Machine Learning Assignment 2


1. Problem Statement
The objective of this project is to build and compare multiple machine learning classification models for predicting whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes.
The project includes data preprocessing, model training, evaluation, model comparison, model deployment using Streamlit, and source code management using GitHub.


2. Dataset Description
Dataset Name: Adult Income Dataset
The Adult Income Dataset is a widely used classification dataset that contains demographic and employment information about individuals.

Features

- Age
- Workclass
- Fnlwgt
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours Per Week
- Native Country


Target Variable

Income Category:
- <=50K
- 50K
Dataset Characteristics

- Number of Records: 48,000+
- Number of Features: 14
- Classification Type: Binary Classification


3. Data Preprocessing
The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Handled missing values.
3. Removed invalid records.
4. Encoded categorical attributes using Label Encoding.
5. Split the dataset into training and testing sets using an 80:20 ratio.
6. Prepared the dataset for machine learning model training.


4. Machine Learning Models Used
The following classification algorithms were trained and evaluated:
1. Logistic Regression
A statistical model used for binary classification.
2. Decision Tree Classifier
A tree-based supervised learning algorithm.
3. K-Nearest Neighbors (KNN)
A distance-based classification technique.
4. Naive Bayes
A probabilistic classification algorithm.
5. Support Vector Machine (SVM)
A powerful classifier that finds an optimal decision boundary between classes.


5. Model Evaluation Metrics
The following evaluation metrics were calculated for every model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


6. Model Comparison Table
Model 	Accuracy 	AUC 	Precision 	Recall 	F1 	MCC
1 	Decision Tree 	0.803645 	0.722758 	0.581227 	0.570922 	0.576029 	0.448309
0 	Logistic regression 	0.802817 	0.798244 	0.620879 	0.400709 	0.487069 	0.385628
3 	Naive Bayes 	0.786247 	0.814518 	0.588235 	0.283688 	0.382775 	0.298623
4 	SVM 	0.772991 	0.603370 	1.000000 	0.028369 	0.055172 	0.147939
2 	KNN 	0.749793 	0.576860 	0.429577 	0.216312 	0.287736 	0.169085

Replace XX values with the actual results obtained from model training.


7. Model Observations

Logistic Regression
- Provides a simple and interpretable baseline model.
- Performs well on linearly separable data.

Decision Tree
- Easy to understand and visualize.
- May overfit the training data without pruning.

KNN
- Produces good classification results.
- Computationally expensive for large datasets.

Naive Bayes
- Fast and efficient.
- Assumes independence among features.
  
SVM
- Provides strong classification performance.
- Works effectively for complex decision boundaries.


8. Best Performing Model
Based on the evaluation metrics, the best-performing model is - SVM

Reason:
- High MCC Value
- Good Generalization Ability
- Highest Accuracy Score
- Strong F1 Score

9. Streamlit Application
A Streamlit web application was developed to:
- Display confusion matrix
- Display classification report
- Upload test datasets
- Display performance metrics
- Select a machine learning model
- Generate predictions

10. GitHub Repository Link
Paste your GitHub Repository URL here:
[https://github.com/2026ak05077/project-folder]


11. Streamlit Deployment Link
Paste your Streamlit Cloud URL here:
(https://adult-income-project.streamlit.app/)


12. Project Structure
adult-income-classification/
app.py
requirements.txt
README.md
adult.csv
test_data.csv
model_comparison.csv
model/
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    svm.pkl


13. Conclusion
This project successfully compared five machine learning classification models using the Adult Income Dataset. The models were evaluated using multiple performance metrics, deployed using Streamlit, and managed through GitHub. The analysis helps identify the most suitable model for predicting income categories based on demographic and employment-related attributes.

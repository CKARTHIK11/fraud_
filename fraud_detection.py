import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
import os
from scipy.stats import randint as sp_rand_int

# Load data
@st.cache_data
def load_data(nRowsRead):
    df = pd.read_csv('creditcard.csv', delimiter=',', nrows=nRowsRead)
    return df

nRowsRead = 1000
df1 = load_data(nRowsRead)

# Display the dataset
st.write(f'There are {df1.shape[0]} rows and {df1.shape[1]} columns')
st.write(df1.head())

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow
    plt.figure(figsize=(6 * nGraphPerRow, 8 * nGraphRow))
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout()
    st.pyplot(plt)

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna(axis='columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        st.write(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(figsize=(graphWidth, graphWidth))
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title('Correlation Matrix', fontsize=15)
    st.pyplot(plt)

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number]) # keep only numerical columns
    df = df.dropna(axis='columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate(f'Corr. coef = {corrs[i, j]:.3f}', (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    st.pyplot(plt)

# Display plots
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 27, 10)

# Prepare and split data
df = pd.read_csv('creditcard.csv', delimiter=',')
X = df.iloc[:, 1:30]
y = df.iloc[:, 30:31]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)

# Print confusion matrix
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))

# Print scores
def print_scores(y_t, y_p):
    st.write(f'Accuracy  :{accuracy_score(y_t, y_p):.2f}')
    st.write(f'Balanced  :{balanced_accuracy_score(y_t, y_p):.2f}')
    st.write(f'F1        :{f1_score(y_t, y_p):.2f}')
    st.write(f'Precision :{precision_score(y_t, y_p):.2f}')
    st.write(f'Recall    :{recall_score(y_t, y_p):.2f}')
    st.write(f'roc auc   :{roc_auc_score(y_t, y_p):.2f}')
    st.write(f'pr)auc    :{average_precision_score(y_t, y_p):.2f}')

print_scores(y_test, y_pred)

# Cross-validation scores
st.write('Cross-validation scores:')
clf = RandomForestClassifier(n_jobs=-1, n_estimators=10, verbose=1)
cvs = cross_val_score(clf, X=X_train, y=y_train, scoring='average_precision')
st.write(f'Using {len(cvs)} trials:\n {cvs}')
st.write(f'Average:{np.mean(cvs)}')

# Cross-validation with multiple scoring metrics
st.write('Cross-validation with multiple scoring metrics:')
clf = RandomForestClassifier(n_jobs=-1, verbose=0)
cvs = cross_validate(clf, X=X_train, y=y_train, scoring=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc'])
for k, v in cvs.items():
    st.write(f'{k.replace("test_", ""):23}{v}   Avg:{"":4}{np.mean(v):.2f}')

# Cross-validation with 10-fold
st.write('Cross-validation with 10-fold:')
clf = RandomForestClassifier(n_jobs=-1, n_estimators=10, verbose=0)
cvs = cross_validate(clf, X=X_train, y=y_train.values.ravel(), cv=10, scoring=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc'])
k_formatter = lambda k: k.replace('test_', '')         # formatter for striping out the test prefix from our CV score names
v_formatter = lambda val: str(val)[:6].ljust(6, '0')  # formatter for displaying our values to 4 significant digits. 

for k, v in cvs.items():
    v_print = ', '.join(map(v_formatter, v))
    st.write(f'{k_formatter(k):23} {v_print}     Avg:  {np.mean(v):.4f}    SDev:  {np.std(v):.4f}')

# GridSearchCV
st.write('GridSearchCV:')
param_grid = {'max_depth': [5, 15, None],
              'max_features': [None, 'sqrt'],
              'n_estimators': [100, 500],
              'min_samples_split': [2, 3, 5]}
clf = RandomForestClassifier(n_jobs=-1, verbose=1, oob_score=True)
clf_cv = GridSearchCV(clf, param_grid, scoring="average_precision", n_jobs=-1, verbose=1)
clf_cv.fit(X_train, y_train)
st.write(clf_cv.best_params_)

# RandomizedSearchCV
st.write('RandomizedSearchCV:')
param_grid = {'max_depth': sp_rand_int(5, 30),
              'max_features': sp_rand_int(5, 30),
              'n_estimators': sp_rand_int(100, 500),
              'min_samples_split': sp_rand_int(2, 5)}
clf = RandomForestClassifier(n_jobs=-1, verbose=1, oob_score=True)
clf_cv = RandomizedSearchCV(clf, param_grid, scoring="average_precision", n_jobs=-1, verbose=1, n_iter=10)
clf_cv.fit(X_train, y_train)
st.write(clf_cv.best_params_)

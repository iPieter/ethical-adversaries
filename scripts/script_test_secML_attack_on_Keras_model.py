import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.path.join("..", "data", "csv", "scikit", "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))

df_binary = df[(df["race"] == "Caucasian") | (df["race"] == "African-American")]

del df_binary['c_jail_in']
del df_binary['c_jail_out']

##separated class from the rests of the features
#remove unnecessary dimensions from Y -> only the decile_score remains
Y = df_binary['decile_score']
del df_binary['decile_score']
del df_binary['two_year_recid']
del df_binary['score_text']

S = df_binary['race']
del df_binary['race']

encod = preprocessing.OrdinalEncoder()
encod.fit(df_binary)
X = encod.transform(df_binary)
X = pd.DataFrame(X)
X.columns = df_binary.columns
X.head()

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=42)

from secml.data import CDataset
tr_set_secML = CDataset(X_train,Y_train)
ts_set_secML = CDataset(X_test,Y_test)

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
#from secml.ml.kernel import CKernelRBF
#clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
from secml.ml.kernel.c_kernel_poly import CKernelPoly
clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelPoly())

# Parameters for the Cross-Validation procedure
#xval_params = {'C': [1e-2, 0.1, 1], 'kernel.gamma': [10, 100, 1e3]}
xval_params = {'C': [1e-4, 1e-3, 1e-2, 0.1, 1], 'kernel.gamma': [0.01, 0.1, 1, 10, 100, 1e3], 'kernel.degree': [2, 3, 5]}

# Let's create a 3-Fold data splitter
random_state = 999

from secml.data.splitter import CDataSplitterKFold
xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

# Select and set the best training parameters for the classifier
print("Estimating the best training parameters...")
best_params = clf.estimate_parameters(
    dataset=tr_set_secML,
    parameters=xval_params,
    splitter=xval_splitter,
    metric='accuracy',
    perf_evaluator='xval'
)
print("The best training parameters are: ", best_params)

# Metric to use for training and performance evaluation
from secml.ml.peval.metrics import CMetricAccuracy
metric = CMetricAccuracy()


# Train the classifier
clf.fit(tr_set_secML)
print(clf.num_classifiers)

# Compute predictions on a test set
y_pred = clf.predict(ts_set_secML.X)

# Evaluate the accuracy of the classifier
acc = metric.performance_score(y_true=ts_set_secML.Y, y_pred=y_pred)

print("Accuracy on test set: {:.2%}".format(acc))


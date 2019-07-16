from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import numpy as np
import pandas
import sys

#first row of the csv is considered as the header so raw.data[0] is directly the first data row
raw_data= pandas.read_csv('../data/csv/compas_recidive_two_years_sanitize_age_value_jail_time_decile_score.csv')

nb_line=len(raw_data)
nb_col=raw_data.size/nb_line

if(nb_line != 6172):
	sys.exit("expected number of lines is 6172 while we retrieved "+str(nb_line))

if(nb_col != 17):
	sys.exit("expected number of cols is 17 while we retrieved "+str(nb_col))

##remove some unwanted columns
del raw_data['c_jail_in']
del raw_data['c_jail_out']
del raw_data['score_text']
res_propub = raw_data['two_year_recid']
del raw_data['two_year_recid']

nb_col_reduced=raw_data.size/nb_line
if(nb_col_reduced != nb_col-4):
	sys.exit("expected number of cols after reduction is 13 while we retrieved "+str(nb_col_reduced))

##separated class from the rests of the features
Y=raw_data['decile_score']
del raw_data['decile_score']

encod = preprocessing.OrdinalEncoder()
encod.fit(raw_data)
raw_data_encoded=encod.transform(raw_data)
raw_data_encoded = pandas.DataFrame(raw_data_encoded)

##divide data into train and test
x_tr, x_ts, y_tr, y_ts = train_test_split(raw_data_encoded,Y, train_size=0.66)
idx = x_ts.index
#print(idx) ##as a row
###print column-wise
#for v in idx:
#	print(v)


#train and valid
#loop for different parameters?
#to be extended for multiple iterations
params=[['gini','best',None,10],
['gini','random',None,10]
]

clf=tree.DecisionTreeClassifier()
clf = clf.fit(x_tr,y_tr)

print(clf.score(x_tr,y_tr))

y_pred=clf.predict(x_ts)

#retrieve misclassified_instances
misclassified_samples = x_ts[y_ts != y_pred]
print(len(misclassified_samples))
print(len(x_ts))
print(len(misclassified_samples)/len(x_ts))


for p in params:
	print(p)
	clf.set_params(criterion=p[0],splitter= p[1],max_depth=p[2],min_samples_split=p[3])
	clf = clf.fit(x_tr,y_tr)

	print(clf.score(x_tr,y_tr))

	y_pred=clf.predict(x_ts)

	#retrieve misclassified_instances
	misclassified_samples = x_ts[y_ts != y_pred]
	print(len(misclassified_samples))
	print(len(x_ts))
	print(len(misclassified_samples)/len(x_ts))


	#check whether misclassified instances are better classified w.r.t. Propublica labels (i.e., groundtruth for recidive)?

	#store results of iterations (for check and comparisons)







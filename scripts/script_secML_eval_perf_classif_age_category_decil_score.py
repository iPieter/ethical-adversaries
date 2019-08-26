from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import numpy as np
import pandas
import sys

##load a method from a different script
from script_factory_classifier_secML import create_classif

def parse_args_into_dict(kwargs):
	print(kwargs)
	param_dic={}
	for arg in kwargs:
		splitted = arg.split("=")
		param_dic[splitted[0]] = splitted[1]
	return param_dic

#### Something to do here to extract variability in the pre-process and only call the rest of the script to train and test ML perfs
#first row of the csv is considered as the header so raw.data[0] is directly the first data row
input_path='../data/csv/scikit/'
filename='compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv'
input_file=input_path+filename
raw_data= pandas.read_csv(input_file)

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

#### End pre-processing and variable part

from secml.array import CArray
#from pandas to CData class of secML?
raw_data_encoded_secML = CArray(raw_data_encoded.values,copy=True)
from secml.data import CDataset
raw_data_encoded_secML = CDataset(raw_data_encoded_secML,Y)

#split between train and test -> train = 66% of data
n_tr = round(raw_data_encoded_secML.num_samples*.66)
n_ts = raw_data_encoded_secML.num_samples - n_tr

#print(n_tr,n_ts)

##divide data into train and test
from secml.data.splitter import CTrainTestSplit
splitter = CTrainTestSplit(
    train_size=n_tr, test_size=n_ts)
x_tr, x_ts = splitter.split(raw_data_encoded_secML)


##describe different models to learn
from array import array
params = [("SVM",dict(C=0.01,kernel="rbf")),
#("SVM",dict(C=0.01,kernel='linear')),
("SVM",dict(C=0.05,kernel="rbf")),
("SVM",dict(C=0.05,kernel="linear")),
("SVM",dict(C=0.1,kernel="rbf")),
("SVM",dict(C=0.1,kernel="linear")),
("SVM",dict(C=0.5,kernel="rbf")),
("SVM",dict(C=0.5,kernel="linear")),
("SVM",dict(C=1,kernel="rbf")),
("SVM",dict(C=1,kernel="linear")),
("SVM",dict(C=5,kernel="rbf")),
("SVM",dict(C=5,kernel="linear")),
("SVM",dict(C=10,kernel="rbf")),
("SVM",dict(C=10,kernel="linear")),
("DT",dict(criterion="gini",splitter="best")),
("DT",dict(criterion="gini",splitter="random")),
("DT",dict(criterion="entropy",splitter="best")),
("DT",dict(criterion="entropy",splitter="random")),
("DT",dict(criterion="gini",splitter="best",max_depth=5)),
("DT",dict(criterion="gini",splitter="random",max_depth=5)),
("DT",dict(criterion="entropy",splitter="best",max_depth=5)),
("DT",dict(criterion="entropy",splitter="random",max_depth=5)),
("DT",dict(criterion="gini",splitter="best",max_depth=10)),
("DT",dict(criterion="gini",splitter="random",max_depth=10)),
("DT",dict(criterion="entropy",splitter="best",max_depth=10)),
("DT",dict(criterion="entropy",splitter="random",max_depth=10)),
("DT",dict(criterion="gini",splitter="best",max_depth=5,min_samples_split=4)),
("DT",dict(criterion="gini",splitter="random",max_depth=5,min_samples_split=4)),
("DT",dict(criterion="entropy",splitter="best",max_depth=5,min_samples_split=4)),
("DT",dict(criterion="entropy",splitter="random",max_depth=5,min_samples_split=4)),
("DT",dict(criterion="gini",splitter="best",max_depth=10,min_samples_split=4)),
("DT",dict(criterion="gini",splitter="random",max_depth=10,min_samples_split=4)),
("DT",dict(criterion="entropy",splitter="best",max_depth=10,min_samples_split=4)),
("DT",dict(criterion="entropy",splitter="random",max_depth=10,min_samples_split=4)),
("DT",dict(criterion="gini",splitter="best",max_depth=5,min_samples_split=10)),
("DT",dict(criterion="gini",splitter="random",max_depth=5,min_samples_split=10)),
("DT",dict(criterion="entropy",splitter="best",max_depth=5,min_samples_split=10)),
("DT",dict(criterion="entropy",splitter="random",max_depth=5,min_samples_split=10)),
("DT",dict(criterion="gini",splitter="best",max_depth=10,min_samples_split=10)),
("DT",dict(criterion="gini",splitter="random",max_depth=10,min_samples_split=10)),
("DT",dict(criterion="entropy",splitter="best",max_depth=10,min_samples_split=10)),
("DT",dict(criterion="entropy",splitter="random",max_depth=10,min_samples_split=10)),
("DT",dict(criterion="gini",splitter="best",max_depth=5,min_samples_split=25)),
("DT",dict(criterion="gini",splitter="random",max_depth=5,min_samples_split=25)),
("DT",dict(criterion="entropy",splitter="best",max_depth=5,min_samples_split=25)),
("DT",dict(criterion="entropy",splitter="random",max_depth=5,min_samples_split=25)),
("DT",dict(criterion="gini",splitter="best",max_depth=10,min_samples_split=25)),
("DT",dict(criterion="gini",splitter="random",max_depth=10,min_samples_split=25)),
("DT",dict(criterion="entropy",splitter="best",max_depth=10,min_samples_split=25)),
("DT",dict(criterion="entropy",splitter="random",max_depth=10,min_samples_split=25))
]


for p in params:
	print("given params:"+str(p))
	#l_args=parse_args_into_dict(p[1])
	#clf = create_classif(p[0],**l_args)
	clf = create_classif(p[0],**p[1])
	#probably need to set the number of iteration to optimize the classifier's function
	clf = clf.fit(x_tr)
	#print(clf.score(x_tr.X,x_tr.Y))

	y_pred=clf.predict(x_ts.X)

	#retrieve misclassified_instances
	#misclassified_samples = x_ts[y_ts != y_pred]
	misclassified_samples = x_ts.Y != y_pred
	#print(len(misclassified_samples))
	nb_mis=misclassified_samples.shape[0]
	print("nb misclassied instances: "+str(nb_mis))
	print("nb instances in test set: "+str(n_ts))
	print("ratio misclassified instances: "+str(nb_mis/n_ts))

	#save results
	##depending on the kind of ML algorithm, find the right path
	from sklearn.externals import joblib

	print(p[0])
	print("Attention!!!")
	print(p[1])

	output_file="../results/"+p[0]+"/model_"+str(filename)
	keys = p[1].keys()
	values = p[1].values()
	for k,v in keys,values:
		output_file = output_file+"_"+str(k)+"_"+str(v)
#	for k,v in p[1].items()
#		output_file.append("_"+str(k)+"_"+str(v))
#	output_file.append(".txt")
	output_file = output_file+".txt"
	clf.save(output_file)
#	joblib.dump(clf,output_file)

	print("YESSSSSSSSSSSSSSSSSSSSSSSSSSS!!!!")

#	output_file_perf="../results/"+str(p[0])+"/perf_"+str(filename)+str(p[1])
	output_file_perf = output_file.replace("model_","perf_")
	f=open(output_file_perf,"w+")
	f.write("given params:"+str(p))
	#f.write("classif score training: "+ str(clf.score(x_tr,y_tr)))
	f.write("nb misclassied instances: "+str(nb_mis))
	#f.write("model classif error on test: "+ str(len(misclassified_samples)))
	f.write("nb instances in test set: "+str(n_ts))
	#f.write("size of test set: "+ str(len(x_ts)))
	f.write("ratio misclassified instances: "+str(nb_mis/n_ts))
	#f.write("percentage of misclassification: "+ str(len(misclassified_samples)/len(x_ts)))
	


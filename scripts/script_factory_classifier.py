from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LinearRegression

import sys


def create_SVM(**kwargs):
	return svm.SVC(**kwargs)

def create_decision_tree(**kwargs):
	return tree.DecisionTreeClassifier(**kwargs)

def create_linear_model(**kwargs):
	classif=LinearRegression(**kwargs)
	return classif


def create_classif(clf_algo, **kwargs):
	if(clf_algo == "SVM"):
		clf = create_SVM(**kwargs)
	elif (clf_algo == "DT"):
		clf = create_decision_tree(**kwargs)
	else:
		clf = create_linear_model(**kwargs)

	print (clf.get_params())
	return clf

def parse_args_into_dict(kwargs):
	param_dic={}
	for arg in kwargs:
		splitted = arg.split("=")
		param_dic[splitted[0]] = splitted[1]
	return param_dic

def main():
	print(len(sys.argv))
	for i in range(0,len(sys.argv)):
		print(sys.argv[i])

	if(len(sys.argv) > 2):
		l_args=parse_args_into_dict(sys.argv[2:])
		create_classif(sys.argv[1], **l_args)
	else:
		create_classif(sys.argv[1])


if __name__== "__main__":
	main()

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers import CClassifierDecisionTree
from secml.ml.classifiers.c_classifier_logistic import CClassifierLogistic
from secml.ml.kernel import CKernelRBF
from secml.ml.kernel import CKernelLinear



clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())

import sys


def create_SVM(**kwargs):
	print("in function, args:")
	print( kwargs)
	return CClassifierMulticlassOVA(CClassifierSVM,**kwargs)

def create_decision_tree(**kwargs):
	print("in function, args:")
	print( kwargs)
	return CClassifierDecisionTree(**kwargs)

def create_linear_model(**kwargs):
	 return CClassifierLogistic(**kwargs)


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

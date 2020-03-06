from secml.ml.features.normalization import CNormalizerMinMax
from secml.array import CArray
from secml.figure import CFigure

def plot_loss_after_attack(evasAttack):

	"""
	This function plots the evolution of the loss function of the surrogate classifier
	after an attack is performed.
	The loss function is normalized between 0 and 1.
	It helps to know whether parameters given to the attack algorithm are well tuned are not;
	the loss should be as minimal as possible.
	The script is inspired from https://secml.gitlab.io/tutorials/11-ImageNet_advanced.html#Visualize-and-check-the-attack-optimization
	"""
	n_iter = evasAttack.x_seq.shape[0]
	itrs = CArray.arange(n_iter)

	# create a plot that shows the loss during the attack iterations
	# note that the loss is not available for all attacks
	fig = CFigure(width=10, height=4, fontsize=14)

	# apply a linear scaling to have the loss in [0,1]
	loss = evasAttack.f_seq
	if loss is not None:
		loss = CNormalizerMinMax().fit_transform(CArray(loss).T).ravel()
		fig.subplot(1, 2, 1)
		fig.sp.xlabel('iteration')
		fig.sp.ylabel('loss')
		fig.sp.plot(itrs, loss, c='black')

	fig.tight_layout()
	fig.show()

# Gradient reversal
The relevant code of the gradient reversal method presented by Adel et al. (2019), Raff et al. (2018) and Ganin et al. (2017). 

## Relevant files
`gradient_reversal.py` contains the main model. Pretrained models are available for COMPAS as `naive_model.h5` and `unbiased_model.h5`. These can be used by:


```python
gr_naive = GradientReversalModel()
gr_naive.load_trained_model(path=naive_model.h5, hp_lambda=0)
Y_pred_n = gr_naive.predict(X_test)
```

A demo notebook illustrating the method is provided in `adversarial_fairness.ipynb`.

Utility scripts are provided in:

- `bayesian_model.py`: For the evaluation of fairness metrics in a compacter form.
- `plot_confusion_matrix.py`: To plot confusion matrices...


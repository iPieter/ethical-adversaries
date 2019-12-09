class BayesianModel:
    """
    Utility class that calculates the probability $$P(X | Y)$$ of
    (sets of) the random variables $X,Y$ with the (set of) random
    variables $Y$ in the conditioning set.
    
    This class should be first initialized with a pandas dataframe,
    afterwards the conditional probability can be calculated as follows:
    
    ```python
    Model(df).P(x = 1).given(y=1)
    ```
    
    Lambdas can also be used, for other options than equalities:
    
    ```python
    Model(df).P(x = lambda x: x>5).given(y=1)
    ```
    """
    def __init__(self, df):
        self._df = df

    def P(self, **kwargs):
        """
        Declares the random variables from the set `kwargs`.
        """
        self._variables = kwargs
        return self

    def given(self, **kwargs):
        """
        Calculates the probability on a finite set of samples with `kwargs` in the
        conditioning set. 
        """
        self._given = kwargs
        
        # Here's where the magic happens
        prior = True
        posterior = True
        
        for k in self._variables:
            if type(self._variables[k]) == type(lambda x:x):
                posterior = posterior & (self._df[k].apply(self._variables[k]))
            else:
                posterior = posterior & (self._df[k] == self._variables[k])

        
        for k in self._given:
            if type(self._given[k]) == type(lambda x:x):
                prior = prior & (self._df[k].apply(self._given[k]))
                posterior = posterior & (self._df[k].apply(self._given[k]))
            else:
                prior = prior & (self._df[k] == self._given[k])
                posterior = posterior & (self._df[k] == self._given[k])
        return posterior.sum()/prior.sum()
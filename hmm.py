import probtorch.model as pm
import probtorch.distributions as pd

import combinators

# class EnsembleHMM(pd.Model):
#     """A HMM with shared parameters for multiple time series"""

#     def __init__(self, num_states):

#     def forward(self):
#         hmm = self.init_hmm(dataset)
#         return pm.Map(hmm)

class HmmInit(combinators.Model):
    pass

class HmmTrans(combinators.Model):
    pass

class HmmLikelihood(combinators.Conditionable):
    pass


class HmmGlobals(combinators.Model):
    def _init__(self, num_states):
        ...

    def forward(self, data, states=None):
        ...
        return data, likelihood, state_init, state_trans


class HmmStates(combinators.Model):
    def __init__(self, num_states):
        ...

    def forward(self, data, likelihood, state_init, state_trans):
        ...
        return states


hmm_globals = HmmGlobals(num_states, name='globals')
hmm_states = HmmStates(num_states, name='states')
hmm = pm.Chain(hmm_globals, hmm_locals, name='hmm') # ToDo: what is syntax for ensuring data is input to both globals and locals
ensemble_hmm = pm.Map(hmm, name='ensemble')

for batch in dataset:
    (likelihood, state_init, state_trans), states = infer(ensemble_hmm, batch)

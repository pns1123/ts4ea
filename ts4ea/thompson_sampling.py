import numpy as np
import pandas as pd

from configuration import ConfigurationEncoder  # , config_encoder
from dataclasses import asdict
from explainer import LIMEConfig, RenderConfig
from itertools import product
from sklearn.linear_model import LogisticRegression
from scipy import stats, optimize


def product_dicts(levels: dict):
    return [
        {key: val for key, val in zip(levels.keys(), prod)}
        for prod in product(*[val for _, val in levels.items()])
    ]


class ThompsonSampler:
    def __init__(self, config_encoder: ConfigurationEncoder):
        self.observations = []
        self.config_encoder = config_encoder

        reference_param_full = {**asdict(LIMEConfig()), **asdict(RenderConfig())}
        reference_param_restr = {
            key: reference_param_full[key]
            for key in self.config_encoder.decode(self.config_encoder.sample_feature())
        }
        # only categorical variables considered for now!
        config_enum = product_dicts(self.config_encoder.categorical_variables)
        config_design_matrix = [
            self.config_encoder.encode(config) for config in config_enum
        ]
        ref_indicator = [
            1 if config == reference_param_restr else 0 for config in config_enum
        ]
        self.X_ref = np.array(config_design_matrix)
        self.y_ref = np.array(ref_indicator)

    def sample_model(self, X: np.ndarray, y: np.ndarray):
        """
        X.shape = (n, m)
        y.shape = (n,)
        """
        X_extended = np.concatenate([self.X_ref, X], axis=0)
        y_extended = np.concatenate([self.y_ref, y], axis=0)
        n_row, n_col = X_extended.shape
        bootstrap_n = n_row
        bootstrap_indices = (
            pd.DataFrame({"y": y_extended})
            .groupby("y")
            .sample(n=bootstrap_n//2+1, replace=True)
            .index.values
        )
        clf = LogisticRegression(random_state=1, max_iter=1000, penalty="l2").fit(
            X_extended[bootstrap_indices, :], y_extended[bootstrap_indices]
        )
        return clf.coef_.squeeze()
        # n = self.config_encoder.categorical_offset[-1] + len(self.config_encoder.numerical_variables)
        # return stats.multivariate_normal(np.zeros(n), np.eye(n)).rvs()

    def select_arm(self, theta):
        """
        input: model coefficient vector, constraints
        output: maximizer
        """
        A, b_l, b_u, integrality = self.config_encoder.milp_constraints()
        constraints = optimize.LinearConstraint(A, b_l, b_u)
        # beware: milp minimization solver -> -theta
        opt_res = optimize.milp(
            c=-theta, constraints=constraints, integrality=integrality
        )
        return self.config_encoder.decode(opt_res.x)



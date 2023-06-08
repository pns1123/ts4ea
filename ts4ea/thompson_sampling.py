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

    @staticmethod
    def amp_update(mu_prior, sigma2_prior, x, y, beta=1):
        def v(t):
            return stats.norm.pdf(t) / stats.norm.cdf(t)

        def w(t):
            return v(t) * (v(t) + t)

        Sigma2 = beta**2 + np.inner(sigma2_prior, x)
        scaled_ip = y * np.inner(x, mu_prior) / np.sqrt(Sigma2)
        mu_posterior = mu_prior + y / np.sqrt(Sigma2) * x * v(scaled_ip)
        sigma2_posterior = sigma2_prior * (
            np.ones(len(sigma2_prior)) - 1 / Sigma2 * x * sigma2_prior * w(scaled_ip)
        )
        return mu_posterior, sigma2_posterior

    @staticmethod
    def sample_model(mu, sigma2):
        """
        mu.shape = (n,)
        sigma2.shape = (n,)
        """
        return stats.multivariate_normal(mu, np.diag(sigma2)).rvs()

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


if __name__ == "__main__":
    categorical_variables = {
        "segmentation_method": [
            "felzenszwalb",
            "slic",
        ],  # , "quickshift", "watershed"],
        "negative": [None, "darkblue"],
        "coverage": [0.15, 0.5, 0.85],
        "opacity": [0.15, 0.5, 0.85],
    }

    # numerical_variables = {"coverage": (0, 1), "opacity": (0, 1)}
    numerical_variables = {}

    config_encoder = ConfigurationEncoder(categorical_variables, numerical_variables)
    n_var = config_encoder.categorical_offset[-1] + len(
        config_encoder.numerical_variables
    )

    thompson_sampler = ThompsonSampler(config_encoder=config_encoder)

    # default_params = {**asdict(LIMEConfig()), **asdict(RenderConfig())}
    default_params = {
        "segmentation_method": "slic",
        "negative": None,
        "coverage": 0.5,
        "opacity": 0.5,
    }
    target_params = {
        "segmentation_method": "slic",
        "negative": "darkblue",
        "coverage": 0.85,
        "opacity": 0.85,
    }
    print(tuple(config_encoder.encode(target_params)))
    print(tuple(config_encoder.encode(default_params)))
    print(
        tuple(config_encoder.encode(target_params))
        <= tuple(config_encoder.encode(default_params))
    )
    mu_prior = config_encoder.encode(default_params)
    sigma2_prior = 5 * np.ones(len(mu_prior))
    print(mu_prior)
    print(sigma2_prior)
    n_iter = 50
    n_samples = 100
    y_late_record = []
    y_early_record = []
    for _ in range(n_samples):
        y_record = []
        mu_prior = config_encoder.encode(default_params)
        sigma2_prior = 5 * np.ones(len(mu_prior))
        for _ in range(n_iter):
            theta = thompson_sampler.sample_model(mu_prior, sigma2_prior)
            config = thompson_sampler.select_arm(theta)
            y = (
                1
                if tuple(config_encoder.encode(config))
                <= tuple(config_encoder.encode(default_params))
                else -1
            )
            y_record.append(y)
            mu_prior, sigma2_prior = thompson_sampler.amp_update(
                mu_prior, sigma2_prior, config_encoder.encode(config), y
            )
        y_early_record.append(np.mean(y_record[:10]))
        y_late_record.append(np.mean(y_record[-10:]))
        #print(72 * "-")
        #print(mu_prior)
        #print(sigma2_prior)
        #print(y_record, np.mean(y_record[:8]), np.mean(y_record[8:]))
        #print(target_params)
        #print(thompson_sampler.select_arm(theta))
    print(y_early_record)
    print(y_late_record)
    print(np.mean(y_early_record), np.mean(y_late_record))

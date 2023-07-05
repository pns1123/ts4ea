import numpy as np

from dataclasses import asdict, dataclass, field
from hashlib import sha256


# HASHING --------------------------------------------------------------------
def config2key(restr_config: dict):

    lime_config_dict = asdict(LIMEConfig())
    for key in restr_config:
        if key in lime_config_dict:
            lime_config_dict[key] = restr_config[key]

    render_config_dict = asdict(RenderConfig())
    for key in restr_config:
        if key in render_config_dict:
            render_config_dict[key] = restr_config[key]


    config = {**lime_config_dict, **render_config_dict}

    return sha256(str((tuple(sorted(config.items())))).encode()).hexdigest()


# EXPLAINER CONFIG -----------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class LIMEConfig:
    """Class for specifying LIME config params"""

    segmentation_method: str = "felzenszwalb"
    num_of_samples: int = 50
    p: float = 0.33


@dataclass(frozen=True, kw_only=True)
class RenderConfig:
    """Class for specifying LIME config params"""

    coverage: float = 0.15
    opacity: float = 0.5
    positive: str = "violet"
    negative: str = None


class ConfigurationEncoder:
    def __init__(
        self,
        categorical_variables: dict = {},
        numerical_variables: dict = {},
    ):
        self.categorical_variables = categorical_variables
        self.categorical_encoder = {
            key: {level: index for index, level in enumerate(levels)}
            for key, levels in categorical_variables.items()
        }

        self.categorical_decoder = {
            key: {index: level for index, level in enumerate(levels)}
            for key, levels in categorical_variables.items()
        }

        self.categorical_offset = self._padded_cumsum(
            [len(encoder) for _, encoder in self.categorical_encoder.items()]
        )

        self.numerical_variables = numerical_variables

    def sample_feature(self):
        sampled_cat_var = {
            var_name: np.random.choice(level)
            for var_name, level in self.categorical_variables.items()
        }
        sampled_num_var = {
            var_name: np.random.uniform(l_b, u_b)
            for var_name, (l_b, u_b) in self.numerical_variables.items()
        }
        return self.encode({**sampled_cat_var, **sampled_num_var})



    @staticmethod
    def zero_one_indicator(n: int, j: int):
        arr = np.zeros(n)
        arr[j] = 1
        return arr

    @staticmethod
    def standard_basis_index(u: np.ndarray):
        return np.where(u == 1)[0].item()

    def _one_hot_constraint(self, encoder):
        A = np.concatenate(
            [
                [self.zero_one_indicator(len(encoder), k) for k in range(len(encoder))]
                + [
                    self.zero_one_indicator(
                        len(encoder), [k for k in range(len(encoder))]
                    )
                ]
            ],
            axis=0,
        )
        b_l = np.zeros(A.shape[0])
        b_l[-1] = 1
        b_u = np.ones(A.shape[0])
        return A, b_l, b_u

    def _numerical_constraints(self):
        if self.numerical_variables == {}:
            return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
        else:
            b_l = [
                l if l is not None else -np.inf
                for _, (l, _) in self.numerical_variables.items()
            ]
            b_u = [
                u if u is not None else np.inf
                for _, (_, u) in self.numerical_variables.items()
            ]
            A = np.concatenate(
                [
                    self.zero_one_indicator(len(self.numerical_variables), k)[None, :]
                    for k in range(len(self.numerical_variables))
                ]
            )
            return A, b_l, b_u

    def _gather_constraints(
        self, categorical_constraints: list[dict], numerical_constraints: list
    ):
        """
        categorical constraints: list of lists
        categorical_constraints[0] dict of A
        categorical_constraints[1] dict of b_l
        categorical_constraints[2] dict of b_u
        """
        A_categorical = categorical_constraints[0]
        b_l_categorical = categorical_constraints[1]
        b_u_categorical = categorical_constraints[2]
        A_numerical = numerical_constraints[0]
        b_l_numerical = numerical_constraints[1]
        b_u_numerical = numerical_constraints[2]

        n_rows = self.categorical_offset[-1] + len(A_categorical) + A_numerical.shape[0]
        n_cols = self.categorical_offset[-1] + A_numerical.shape[1]
        A_full = np.zeros((n_rows, n_cols))
        principal_row_indices = self.categorical_offset.copy()
        principal_row_indices += np.array(range(len(principal_row_indices)))

        for k, cat_var_name in enumerate(A_categorical):
            A_full[
                principal_row_indices[k] : principal_row_indices[k + 1],
                self.categorical_offset[k] : self.categorical_offset[k + 1],
            ] = A_categorical[cat_var_name]

        A_full[principal_row_indices[-1] :, self.categorical_offset[-1] :] = A_numerical

        b_l_full = np.concatenate(
            [b_l for _, b_l in b_l_categorical.items()] + [b_l_numerical]
        )
        b_u_full = np.concatenate(
            [b_u for _, b_u in b_u_categorical.items()] + [b_u_numerical]
        )
        return A_full, b_l_full, b_u_full

    def milp_constraints(self):
        constraint_dict = {
            var_name: self._one_hot_constraint(self.categorical_encoder[var_name])
            for var_name, _ in self.categorical_encoder.items()
        }
        A_categorical = {
            var_name: self._one_hot_constraint(encoder)[0]
            for var_name, encoder in self.categorical_encoder.items()
        }
        b_l_categorical = {
            var_name: self._one_hot_constraint(encoder)[1]
            for var_name, encoder in self.categorical_encoder.items()
        }
        b_u_categorical = {
            var_name: self._one_hot_constraint(encoder)[2]
            for var_name, encoder in self.categorical_encoder.items()
        }

        A_numerical, b_l_numerical, b_u_numerical = self._numerical_constraints()

        A, b_l, b_u = self._gather_constraints(
            [A_categorical, b_l_categorical, b_u_categorical],
            [A_numerical, b_l_numerical, b_u_numerical],
        )

        integrality = self.zero_one_indicator(
            A.shape[1], [i for i in range(self.categorical_offset[-1])]
        )

        return A, b_l, b_u, integrality

    @staticmethod
    def _padded_cumsum(arr: np.ndarray):
        cs = np.zeros(len(arr) + 1)
        cs[1:] = np.cumsum(arr).astype(int)
        return cs.astype(int)

    def encode(self, config: dict):
        return np.concatenate(
            [
                self.zero_one_indicator(
                    len(self.categorical_encoder[var_name]),
                    self.categorical_encoder[var_name][config[var_name]],
                )
                for var_name, encoder in self.categorical_encoder.items()
            ]
            + [
                np.array(
                    [config[numeric_var] for numeric_var in self.numerical_variables]
                )
            ]
        )

    def decode(self, features: np.ndarray):
        return {
            **{
                categorical_var_name: self.categorical_decoder[categorical_var_name][
                    self.standard_basis_index(
                        features[
                            self.categorical_offset[i] : self.categorical_offset[i + 1]
                        ]
                    )
                ]
                for i, categorical_var_name in enumerate(self.categorical_encoder)
            },
            **{
                numerical_var_name: features[self.categorical_offset[-1] + i]
                for i, numerical_var_name in enumerate(self.numerical_variables)
            },
        }



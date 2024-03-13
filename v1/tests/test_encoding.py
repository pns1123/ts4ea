import numpy as np

from ts4ea.configuration import ConfigurationEncoder


class TestEncodingA:
    def test_encoding(self):
        categorical_vars = {"A": ["x", "y", "z"]}
        numerical_vars = {"B": (None, 7)}
        config_encoder = ConfigurationEncoder(categorical_vars, numerical_vars)
        test_config = {"A": "y", "B": 3}
        assert (config_encoder.encode(test_config) == np.array([0, 1, 0, 3])).all

    def test_inversion(self):
        categorical_vars = {"A": ["x", "y", "z"]}
        numerical_vars = {"B": (None, 7)}
        config_encoder = ConfigurationEncoder(categorical_vars, numerical_vars)
        test_config = {"A": "y", "B": 3}
        assert config_encoder.decode(config_encoder.encode(test_config)) == test_config


class TestConstraints:
    def test_one_hot_constraint_one_var(self):
        test_cat_var = {"x": ["blue", "green", "red"]}
        config_encoder = ConfigurationEncoder(test_cat_var)
        A, b_l, b_u = config_encoder._one_hot_constraint(
            config_encoder.categorical_encoder["x"]
        )
        assert (A == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])).all
        assert (b_l == np.array([0, 0, 0, 0])).all
        assert (b_u == np.array([1, 1, 1, 1])).all

    def test_one_hot_constraint_two_vars(self):
        test_cat_var = {"x": ["blue", "green"], "y": ["high", "low"]}
        config_encoder = ConfigurationEncoder(test_cat_var)
        constraint_dict = {
            var_name: config_encoder._one_hot_constraint(
                config_encoder.categorical_encoder[var_name]
            )
            for var_name in ["x", "y"]
        }
        print(constraint_dict["x"][0])
        print(constraint_dict["x"][0].shape)
        assert constraint_dict["x"][0].shape == (3, 2)
        assert (constraint_dict["x"][0] == np.array([[1, 0], [0, 1], [1, 1]])).all
        assert (constraint_dict["x"][1] == np.array([0, 0, 0])).all
        assert (constraint_dict["x"][2] == np.array([1, 1, 1])).all

        assert constraint_dict["y"][0].shape == (3, 2)
        assert (constraint_dict["y"][0] == np.array([[1, 0], [0, 1], [1, 1]])).all
        assert (constraint_dict["y"][1] == np.array([0, 0, 0])).all
        assert (constraint_dict["y"][2] == np.array([1, 1, 1])).all

    def test_numerical_constraints(self):
        test_num_var = {"x": (None, None), "y": (0, None), "z": (None, 1), "zz": (0, 1)}
        config_encoder = ConfigurationEncoder(numerical_variables=test_num_var)
        A, b_l, b_u = config_encoder._numerical_constraints()
        assert (
            A == np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ).all
        assert (b_l == np.array([-np.inf, 0, -np.inf, 0])).all
        assert (b_u == np.array([np.inf, np.inf, 1, 1])).all

    def test_gather(self):
        test_cat_var = {"x": ["blue", "green"], "y": ["high", "low"]}
        test_num_var = {"z": (42, 69)}
        config_encoder = ConfigurationEncoder(test_cat_var, test_num_var)
        constraint_dict = {
            var_name: config_encoder._one_hot_constraint(
                config_encoder.categorical_encoder[var_name]
            )
            for var_name in ["x", "y"]
        }
        A_categorical = {
            var_name: config_encoder._one_hot_constraint(encoder)[0]
            for var_name, encoder in config_encoder.categorical_encoder.items()
        }
        b_l_categorical = {
            var_name: config_encoder._one_hot_constraint(encoder)[1]
            for var_name, encoder in config_encoder.categorical_encoder.items()
        }
        b_u_categorical = {
            var_name: config_encoder._one_hot_constraint(encoder)[2]
            for var_name, encoder in config_encoder.categorical_encoder.items()
        }


        (
            A_numerical,
            b_l_numerical,
            b_u_numerical,
        ) = config_encoder._numerical_constraints()
        A, b_l, b_u = config_encoder._gather_constraints(
            [A_categorical, b_l_categorical, b_u_categorical], 
            [A_numerical, b_l_numerical, b_u_numerical]
        )
        assert (
            A
            == np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
        ).all
        assert (b_l == np.array([0, 0, 0, 0, 0, 0, 42])).all
        assert (b_u == np.array([1, 1, 1, 1, 1, 1, 69])).all

import math
import random
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from numpy import ndarray
from numpy.random import default_rng
from statsmodels.stats import correlation_tools


def _build_pseudo_classes(params_dict: Dict[str, Any]) -> ndarray:
    """Create pseudo-classes by shuffling artificial classes.

    The total number of underlying classes equals the total number of artificial classes. The
    underlying classes for the pseudo-class are created exactly like the artificial classes.

    Args:
        params_dict: Parameter dict containing number of pseudo-class features, number of artificial classes,
            their distributions and parameters (see parameters of :func:`generate_artificial_classification_data`).

    Returns:
        Randomly shuffled pseudo-class: Numpy array of the given shape.

    """
    rng = default_rng()
    simulated_classes = []
    number_of_pseudo_class_samples = 0

    # generate normal distributed classes
    for class_number, class_params_dict in params_dict["classes"].items():
        normal_distributed_class = rng.normal(
            size=(
                class_params_dict["number_of_samples"],
                params_dict["number_of_pseudo_class_features"],
            ),
            loc=2 * class_number,  # shift random data to generate different classes
            scale=1,
        )
        simulated_classes.append(normal_distributed_class)

        # assert shape and mean of simulated class
        number_of_pseudo_class_samples += class_params_dict["number_of_samples"]
        if not math.isclose(
            np.mean(normal_distributed_class),
            2 * class_number,
            abs_tol=0.4,
        ):
            print(
                f"INFO: Mean {np.mean(normal_distributed_class)} "
                f"of generated data differs from expected mean {2*class_number}"
                f" within the pseudo class."
            )

    classes = np.concatenate(simulated_classes, axis=0)

    assert classes.shape == (
        number_of_pseudo_class_samples,
        params_dict["number_of_pseudo_class_features"],
    )

    # shuffle classes to finally create pseudo-class
    np.random.shuffle(classes)
    return classes


def _build_single_class(class_params_dict: Dict[str, Any], number_of_relevant_features: int) -> ndarray:

    class_data_np = _generate_normal_distributed_class(class_params_dict, number_of_relevant_features)
    if class_params_dict["distribution"] == "lognormal":
        class_data_np = np.exp(class_data_np)

    # shift class data to expected mode
    class_data_np = class_data_np + class_params_dict["mode"]

    return class_data_np


def _generate_normal_distributed_class(class_params_dict: Dict[str, Any], number_of_relevant_features: int) -> ndarray:
    class_features_list = []

    number_of_normal_distributed_random_features = number_of_relevant_features
    # generate correlated features?
    if len(class_params_dict["correlated_features"]) > 0:
        correlated_features = _generate_correlated_features(class_params_dict)
        assert correlated_features.shape[1] <= number_of_normal_distributed_random_features
        number_of_normal_distributed_random_features -= correlated_features.shape[1]
        assert number_of_normal_distributed_random_features >= 0
        assert correlated_features.shape[0] == class_params_dict["number_of_samples"]

        class_features_list.append(correlated_features)

    # generate normal distributed random data
    if number_of_normal_distributed_random_features > 0:
        rng = default_rng()
        relevant_features = rng.normal(
            loc=0,
            scale=class_params_dict["scale"],
            size=(class_params_dict["number_of_samples"], number_of_normal_distributed_random_features),
        )

        if not math.isclose(np.mean(relevant_features), 0, abs_tol=0.15):
            warnings.warn(
                f"mean of generated data {str(np.mean(relevant_features))} differs from expected mean {str(0)} "
                f"-> Try choosing a smaller scale for a small sample size or accept a deviating mean. "
                f"The current scale is {str(class_params_dict['scale'])}."
            )
        class_features_list.append(relevant_features)

    complete_class_np = np.concatenate(class_features_list, axis=1)
    assert complete_class_np.shape == (class_params_dict["number_of_samples"], number_of_relevant_features)
    return complete_class_np


def _generate_correlated_cluster(
    number_of_features: int,
    number_of_samples: int,
    lower_bound: float,
    upper_bound: float,
) -> ndarray:
    """Generate a block of correlated features.

    Args:
        number_of_features: Number of columns of generated data.
        number_of_samples: Number of rows of generated data.
        lower_bound: Lower bound of the generated correlations.
        upper_bound: Upper bound of the generated correlations.

    Returns:
        Numpy array of the given shape with correlating features in the given range.

    """
    rng = default_rng()

    # generate random matrix to constrain a range of values
    # and specify a starting point
    random_matrix = np.random.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(number_of_features, number_of_features),
    )

    print("generation of correlation matrix ...")
    # first iteration generating correlations to
    # improve the fit of the covariance matrix
    correlation_matrix = correlation_tools.corr_nearest(corr=random_matrix, threshold=1e-15, n_fact=100)

    # change values on the diagonal to 1 to
    # improve the fit of the covariance matrix
    for i in range(0, correlation_matrix.shape[0]):
        correlation_matrix[i, i] = 1

    print("generation of covariant matrix ...")
    # generate the nearest positive semi-definite covariance matrix
    covariance_matrix = correlation_tools.cov_nearest(
        correlation_matrix,
        method="nearest",
        threshold=1e-15,
        n_fact=1000,
        return_all=False,
    )
    print(covariance_matrix)
    print("generation of covariant matrix finished")

    # generate correlated cluster
    covariant_cluster = rng.multivariate_normal(
        mean=np.zeros(number_of_features),
        cov=covariance_matrix,
        size=number_of_samples,
        check_valid="raise",
    )
    print("generation of correlated cluster finished")

    return covariant_cluster


def _generate_correlated_features(class_params_dict: Dict[str, Any]) -> ndarray:
    correlated_feature_clusters = []
    for cluster_params_dict in class_params_dict["correlated_features"].values():
        correlated_feature_cluster = _generate_correlated_cluster(
            number_of_features=cluster_params_dict["number_of_features"],
            number_of_samples=class_params_dict["number_of_samples"],
            lower_bound=cluster_params_dict["correlation_lower_bound"],
            upper_bound=cluster_params_dict["correlation_upper_bound"],
        )
        correlated_feature_clusters.append(correlated_feature_cluster)
    correlated_features = np.concatenate(correlated_feature_clusters, axis=1)
    return correlated_features


def _generate_dataframe(
    data_np: np.ndarray,
    params_dict: Dict[str, Any],
) -> pd.DataFrame:
    """Generate semantic names for the columns of the given DataFrame.

    Args:
        data_np: Numpy array with generated data.
        params_dict: Parameter dict including the number of features per
            class, the number of pseudo-class features and the number of random
            features (see parameters of
            :func:`generate_artificial_classification_data`).

    Returns:
        DataFrame with meaningful named columns.
            - 'label' for the labels
            - `bm` for artificial class feature
            - `pseudo` for pseudo-class feature
            - `random` for random data

    """
    # generate label as first entry
    column_names = ["label"]

    # generate names for artificial biomarkers
    number_of_names = 0
    for class_params_dict in params_dict["classes"].values():
        if class_params_dict["correlated_features"]:
            for cluster_number, cluster_parameter in class_params_dict["correlated_features"].items():
                for feature_number in range(cluster_parameter["number_of_features"]):
                    column_names.append(f"bmc{cluster_number}_{feature_number}")
                    number_of_names += 1

    for column_name in range(params_dict["number_of_relevant_features"] - number_of_names):
        column_names.append(f"bm_{column_name}")

    for column_name in range(params_dict["number_of_pseudo_class_features"]):
        column_names.append(f"pseudo_{column_name}")

    for column_name in range(params_dict["random_features"]["number_of_features"]):
        column_names.append(f"random_{column_name}")

    data_df = pd.DataFrame(data=data_np, columns=column_names)
    data_df = _shuffle_features(data_df, params_dict)
    _save(data_df, params_dict)
    print("data shape: ", data_df.shape)
    return data_df


def _shuffle_features(data_df: pd.DataFrame, params_dict: Dict[str, any]):
    # shuffle all features
    if params_dict["shuffle_features"]:
        column_names = list(data_df.columns[1:])
        random.shuffle(column_names)
        shuffled_column_names = ["label"] + column_names
        data_df = data_df[shuffled_column_names]
        assert data_df.columns[0] == "label"
    return data_df


def _save(data_df, params_dict):
    if "path_to_save_csv" in params_dict.keys() and params_dict["path_to_save_csv"]:
        assert isinstance(params_dict["path_to_save_csv"], str)
        assert params_dict["path_to_save_csv"].endswith(".csv")
        pd.DataFrame(data_df).to_csv(params_dict["path_to_save_csv"], index=False)
        print(f"Data generated successfully and saved in {params_dict['path_to_save_csv']}")

    if "path_to_save_feather" in params_dict.keys() and params_dict["path_to_save_feather"]:
        assert isinstance(params_dict["path_to_save_feather"], str)
        assert params_dict["path_to_save_feather"].endswith(".feather")
        pd.DataFrame(data_df).to_feather(params_dict["path_to_save_feather"], index=False)
        print(f"Data generated successfully and saved in {params_dict['path_to_save_feather']}")

    if "path_to_save_meta_data" in params_dict.keys() and params_dict["path_to_save_meta_data"]:
        assert isinstance(params_dict["path_to_save_meta_data"], str)
        assert params_dict["path_to_save_meta_data"].endswith(".yaml")
        assert isinstance(params_dict, dict)
        # f = open(params_dict["path_to_save_meta_data"], "x")  # throw exception if file exists
        with open(params_dict["path_to_save_meta_data"], "w") as f:
            yaml.dump(params_dict, f)
        print(f"Meta data successfully saved in {params_dict['path_to_save_meta_data']}")


def generate_artificial_classification_data(params_dict: Dict[str, Any]):
    """Generate artificial classification (e.g. biomarker) data.

    Args:
        params_dict: Parameters for the data to generate
                        Example:
                        params_dict = {
                                        "number_of_relevant_features": 12,
                                        "number_of_pseudo_class_features": 2,
                                        "random_features": {"number_of_features": 10, "distribution": "lognormal",
                                                            "scale": 1, "mode": 0},
                                        "classes": {
                                            1: {
                                                "number_of_samples": 15,
                                                "distribution": "lognormal",
                                                "mode": 3,
                                                "scale": 1,
                                                "correlated_features": {
                                                    1: {"number_of_features": 4, "correlation_lower_bound": 0.7,
                                                        "correlation_upper_bound": 1},
                                                    2: {"number_of_features": 4, "correlation_lower_bound": 0.7,
                                                        "correlation_upper_bound": 1},
                                                    3: {"number_of_features": 4, "correlation_lower_bound": 0.7,
                                                        "correlation_upper_bound": 1},
                                                },
                                            },
                                            2: {"number_of_samples": 15, "distribution": "normal", "mode": 1,
                                                "scale": 2, "correlated_features": {}},
                                            3: {"number_of_samples": 15, "distribution": "normal", "mode": -10,
                                                "scale": 2, "correlated_features": {}},
                                        },
                                        "path_to_save_csv": "your_path_to_save.csv",
                                        "path_to_save_feather": "",
                                        "path_to_save_meta_data": "your_path_to_save_params_dict.yaml",
                                        "shuffle_features": False,
                                      }


                        "number_of_relevant_features": Total number of features (columns) to generate
                                                        for each artificial class.
                        "number_of_pseudo_class_features": Number of pseudo-class features.
                            The underlying classes correspond to the selected number of classes and follow a normal
                            distribution. Shifted modes of the generated artificial classes equal two times
                            the class number. All samples of the generated classes are randomly shuffled and
                            therefore have no relation to any class label.
                        "random_features": {"number_of_features": Number of randomly generated features.
                                            "distribution": "lognormal" or "normal",
                                            "scale": Standard deviation (spread or “width”) of the distribution.
                                                     Must be non-negative.,
                                            "mode": Mean (“centre”) of the distribution.},
                        "classes":  Parameter dicts for each class to generate. The key equals the class label.
                                    "number_of_samples": 15,
                                    "distribution": "lognormal",
                                    "mode": 3,
                                    "scale": 1,
                                    "correlated_features": Parameter dicts for each cluster of correlated features to
                                                           generate. The key equals the cluster number.
                                                           To generate no clusters insert empty dict.
                                                            "number_of_features": Number of correlated features within
                                                                                  a cluster.
                                                            "correlation_lower_bound": Lower bounds for the correlation
                                                                                       of each cluster of correlated
                                                                                       features within a normal
                                                                                       distributed class.
                                                                                       Default is 0.7.
                                                            "correlation_upper_bound": Upper bounds for the correlation
                                                                                       of each cluster of correlated
                                                                                       features within a normal
                                                                                       distributed class. Default is 1.
                        "path_to_save_csv": "your_path_to_save.csv",
                        "path_to_save_feather": "your_path_to_save.feather",
                        "path_to_save_meta_data": "your_path_to_save_params_dict.yaml",
                        "shuffle_features": False,
                      }
    Returns:
        Generated artificial data as DataFrame.
    """

    # validate input parameters

    # generate relevant features
    classes_list = []
    for class_number, class_params_dict in params_dict["classes"].items():
        # generate label
        label_np = np.full((class_params_dict["number_of_samples"], 1), fill_value=class_number)
        data_class_np = _build_single_class(class_params_dict, params_dict["number_of_relevant_features"])
        labeled_data_class_np = np.concatenate((label_np, data_class_np), axis=1)
        classes_list.append(labeled_data_class_np)

    class_features_np = np.concatenate(classes_list, axis=0)
    assert class_features_np.shape[1] == params_dict["number_of_relevant_features"] + 1  # the label

    # generate random features
    random_features_np = np.random.normal(
        loc=0.0,
        scale=params_dict["random_features"]["scale"],
        size=(class_features_np.shape[0], params_dict["random_features"]["number_of_features"]),
    )
    # shift mode of random features
    random_features_np += params_dict["random_features"]["mode"]

    # generate pseudo class features
    pseudo_class_features_np = _build_pseudo_classes(params_dict)
    assert pseudo_class_features_np.shape == (
        class_features_np.shape[0],
        params_dict["number_of_pseudo_class_features"],
    )

    artificial_data_np = np.concatenate((class_features_np, pseudo_class_features_np, random_features_np), axis=1)
    assert not np.isnan(artificial_data_np).any()
    assert (
        artificial_data_np.shape[1]
        == params_dict["number_of_pseudo_class_features"]
        + params_dict["random_features"]["number_of_features"]
        + params_dict["number_of_relevant_features"]
        + 1  # the label
    )
    data_df = _generate_dataframe(data_np=artificial_data_np, params_dict=params_dict)
    return data_df

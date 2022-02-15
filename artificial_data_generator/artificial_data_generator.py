# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Generator for artificial data.

Can be used as baseline for benchmarking and the development of new methods.
For example, simulation of biomarker data from high-throughput experiments.
"""
import math
import random
import warnings
from numbers import Number
from typing import Any, Dict, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from numpy import ndarray
from numpy.random import default_rng
from sklearn.utils import shuffle
from statsmodels.stats import correlation_tools


def _generate_normal_distributed_class(
    label: float,
    number_of_samples: int,
    number_of_features: int,
    scale: float,
) -> ndarray:
    """Generate artificial normal distributed class (e.g. simulate the samples from healthy patients).

    Args:
        label: Label for the generated artificial class.
        number_of_samples: Number of rows of generated data.
        number_of_features: Number of columns of generated data.
        scale: Scale of the normal distribution.

    Returns:
        Normal distributed data of the given shape and parameters with the given label in the first column.

    """
    # generate labels
    label_vector = np.full((number_of_samples, 1), label)

    # generate data
    rng = default_rng()
    features = rng.normal(
        loc=0,
        scale=scale,
        size=(number_of_samples, number_of_features),
    )
    print("scale", scale)
    if not math.isclose(np.mean(features), 0, abs_tol=0.15):
        warnings.warn(
            "mean of generated data: "
            + str(np.mean(features))
            + " differs from expected mean: "
            + str(0)
            + " -> Try choosing a smaller scale for a "
            "small sample size or accept a deviating mean."
            " The current scale is " + str(scale) + "."
        )

    normal_distributed_data = np.hstack((label_vector, features))
    return normal_distributed_data


def _generate_pseudo_class(params_dict: Dict[str, Any]) -> ndarray:
    """Create a pseudo-class by shuffling artificial classes.

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

    # generate lognormal distributed classes
    for _ in range(params_dict["number_of_lognormal_distributed_classes"]):

        # shift random lognormal data to generate different classes
        lognormal_distributed_class = rng.lognormal(
            size=(
                params_dict["number_of_samples_per_class"],
                params_dict["number_of_pseudo_class_features"],
            )
        )
        if not math.isclose(
            np.mean(np.log(lognormal_distributed_class)),
            0,
            abs_tol=0.4,
        ):
            print(
                f"INFO: Mean {np.mean(np.log(lognormal_distributed_class))} "
                f"of the underlying normal distribution of the generated data "
                f"differs from expected mean 0"
                f" within the pseudo class."
            )
        simulated_classes.append(lognormal_distributed_class)

    # generate normal distributed classes
    for _ in range(params_dict["number_of_normal_distributed_classes"]):
        # shift random data to generate different classes
        normal_distributed_class = rng.standard_normal(
            size=(
                params_dict["number_of_samples_per_class"],
                params_dict["number_of_pseudo_class_features"],
            )
        )
        if not math.isclose(
            np.mean(normal_distributed_class),
            0,
            abs_tol=0.4,
        ):
            print(
                f"INFO: Mean {np.mean(normal_distributed_class)} "
                f"of generated data differs from expected mean 0"
                f" within the pseudo class."
            )
        simulated_classes.append(normal_distributed_class)

    shifted_simulated_classes, _ = _shift_all_classes(simulated_classes, params_dict)
    classes = np.concatenate(shifted_simulated_classes, axis=0)

    assert classes.shape == (
        (
            (
                params_dict["number_of_lognormal_distributed_classes"]
                + params_dict["number_of_normal_distributed_classes"]
            )
            * params_dict["number_of_samples_per_class"]
        ),
        params_dict["number_of_pseudo_class_features"],
    )

    # shuffle classes to finally create pseudo-class
    pseudo_class = shuffle(pd.DataFrame(classes))
    return np.array(pseudo_class)


def _generate_normal_distributed_correlated_block(
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

    # generate intra correlated class
    covariant_block = rng.multivariate_normal(
        mean=np.zeros(number_of_features),
        cov=covariance_matrix,
        size=number_of_samples,
        check_valid="raise",
    )
    print("correlated class generation finished")
    _visualize_correlations(covariant_block)

    return covariant_block


def _visualize_correlations(data: Union[ndarray, pd.DataFrame]) -> None:
    """Visualize correlations.

    Args:
        data: DataFrame or numpy array where each column equals a class.

    """
    # convert numpy array to DataFrame
    data_df = pd.DataFrame(data)

    # generate feature names
    column_names = []
    for column_name in data_df.columns:
        column_names.append("feature_" + str(column_name))
    data_df.columns = column_names

    sns.set_theme(style="white")
    corr = data_df.corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".1g")
    pyplot.show()
    # pyplot.savefig('C:/Users/sma19/Pictures/
    # correlation_FS/healthy_{}.png'.format(data_name))


def _visualize_distributions(data: Union[ndarray, pd.DataFrame]) -> None:
    """Visualize the distribution of different classes.

    Args:
        data: DataFrame or numpy array where each column equals a class.

    """
    # convert numpy array to DataFrame
    data_df = pd.DataFrame(data)

    # generate column names
    column_names = []
    for column_name in data_df.columns:
        column_names.append("element_" + str(column_name))
    data_df.columns = column_names

    sns.set(color_codes=True)
    sns.displot(data=data_df, kde=True)
    pyplot.show()


def _generate_column_names(
    data_df: pd.DataFrame,
    params_dict: Dict[str, Any],
) -> pd.DataFrame:
    """Generate semantic names for the columns of the given DataFrame.

    Args:
        data_df: DataFrame with unnamed columns.
        params_dict: Parameter dict including the number of features per
            class, the number of pseudo-class features and the number of random
            features (see parameters of
            :func:`generate_artificial_classification_data`).

    Returns:
        DataFrame with meaningful named columns.
            - `bm` for artificial class feature
            - `pseudo` for pseudo-class feature
            - `random` for random data

    """
    # generate label as first entry
    column_names = ["label"]

    # generate names for artificial biomarkers
    for column_name in range(params_dict["number_of_features_per_class"]):
        column_names.append("bm_" + str(column_name))

    for column_name in range(params_dict["number_of_pseudo_class_features"]):
        column_names.append("pseudo_" + str(column_name))

    for column_name in range(params_dict["number_of_random_features"]):
        column_names.append("random_" + str(column_name))

    data_df.columns = column_names
    return data_df


def _shift_all_classes(classes_list: List[ndarray], params_dict: Dict[str, Any]):
    """Shift the locale of all classes.

    Args:
        classes_list: List of classes as numpy arrays.
        params_dict: Dict including the shift values for all classes.

    Returns:
        List of shifted classes.

    """
    classes_df = pd.DataFrame()
    shifted_classes = []

    # shift all classes
    for generated_class, shift in zip(classes_list, params_dict["all_shifts"]):
        #  shift class data and exclude the label from shifting
        label = generated_class[:, 0].reshape(-1, 1)
        shifted_class_data = generated_class[:, 1:] + shift
        classes_df["mean_" + str(shift)] = shifted_class_data.flatten()

        labeled_shifted_class = np.hstack((label, shifted_class_data))
        assert labeled_shifted_class[:, 0].all() == label.all()
        shifted_classes.append(labeled_shifted_class)

    return shifted_classes, classes_df


def _transform_normal_to_lognormal(classes_list: List[ndarray]) -> List[ndarray]:
    """Transform normal distributed data to lognormal distributions.

    Args:
        classes_list: List of classes as numpy arrays.

    Returns:
        List of classes as numpy arrays transformed to lognormal distributions.

    """
    lognormal_distributed_classes = []

    # transform normal distributions to lognormal distributions
    for generated_class in classes_list:
        #  transform class data excluding the label
        label = generated_class[:, 0].reshape(-1, 1)
        lognormal_distributed_class = np.exp(generated_class[:, 1:])
        labeled_lognormal = np.concatenate((label, lognormal_distributed_class), axis=1)
        lognormal_distributed_classes.append(labeled_lognormal)

    return lognormal_distributed_classes


def _generate_normal_distributed_classes(
    labels: List[float],
    meta_data_dict: Dict[str, Union[List[List[str]], List[str], int]],
    number_of_features: int,
    number_of_classes: int,
    number_of_features_per_correlated_block: Union[ndarray, List[List[int]]],
    number_of_samples_per_class: int,
    scales: Union[List[float], ndarray],
    lower_bounds_for_correlations: Union[List[float], ndarray],
    upper_bounds_for_correlations: Union[List[float], ndarray],
) -> List[ndarray]:
    """Generate artificial classes with the given parameters as numpy arrays.

    Args:
        labels: Labels of the classes to be generated.
        meta_data_dict: Dict to store metadata for the correlated features within the artificial classes.
        number_of_features: Number of features (columns) per class.
        number_of_classes: Nunmber of classes to be generated.
        number_of_features_per_correlated_block: Number of features within each block
            of correlated features for each class.
        number_of_samples_per_class: Number of samples (rows) per class.
        scales: Scales of the respective class distributions.
        lower_bounds_for_correlations: Lower bounds for the correlations of the separate blocks
            of correlated features within the respective generated classes.
        upper_bounds_for_correlations: Upper bounds for the correlations of the separate blocks
            of correlated features within the respective generated classes.

    Returns:
        List of generated normal distributed classes of the given shape as numpy arrays.
    """
    classes = []

    assert len(scales) == number_of_classes
    assert len(labels) == number_of_classes
    # simulation of intraclass correlation
    if number_of_features_per_correlated_block is not None:
        # generate intraclass correlated classes
        for i, label in enumerate(labels):
            assert len(number_of_features_per_correlated_block[i]) > 0

            # generate blocks of correlated features
            blocks = [
                _generate_normal_distributed_correlated_block(
                    number_of_features=number_of_features_per_correlated_block[i][j],
                    number_of_samples=number_of_samples_per_class,
                    lower_bound=lower_bounds_for_correlations[i],
                    upper_bound=upper_bounds_for_correlations[i],
                )
                for j in range(len(number_of_features_per_correlated_block[i]))
            ]
            # generate class feature names
            class_features = []
            for block in blocks:
                block_features = []
                for feature_count in range(block.shape[1]):
                    block_features.append("corr_" + str(feature_count))
                assert len(block_features) == block.shape[1]
                class_features.append(block_features)
            meta_data_dict["class_" + str(label) + "_blocks"] = len(number_of_features_per_correlated_block[i])
            meta_data_dict["class_" + str(label)] = class_features

            generated_class = np.concatenate(blocks, axis=1)

            _visualize_correlations(generated_class)

            # generate uncorrelated features
            if generated_class.shape[1] < number_of_features:
                uncorrelated_features = _generate_normal_distributed_class(
                    label,
                    number_of_samples_per_class,
                    number_of_features=number_of_features - generated_class.shape[1],
                    scale=scales[i],
                )
                unlabeled_uncorrelated_features = uncorrelated_features[:, 1:]
                # exclude the label as it is appended later
                generated_class = np.concatenate((generated_class, unlabeled_uncorrelated_features), axis=1)
                # generate unlabeled uncorrelated class feature names
                for uncorrelated_feature_number in range(unlabeled_uncorrelated_features.shape[1]):
                    class_features.append(["uncorr_" + str(uncorrelated_feature_number)])
                meta_data_dict["class_" + str(label)] = class_features

            assert generated_class.shape[1] == number_of_features
            assert generated_class.shape[0] == number_of_samples_per_class

            # generate names to check the number of generated features
            class_feature_names_array = np.concatenate(
                [
                    np.array(i) for i in meta_data_dict["class_" + str(label)]  # type:ignore
                ]
            )
            assert class_feature_names_array.size == number_of_features, (
                "Number of features "
                + str(number_of_features)
                + " is not equal to size of feature meta data "
                + str(class_feature_names_array.size)
            )

            # generate labels
            label_vector = np.full((number_of_samples_per_class, 1), label)

            labeled_class = np.concatenate((label_vector, generated_class), axis=1)
            assert labeled_class.shape[1] == number_of_features + 1
            classes.append(labeled_class)
        assert len(classes) == len(number_of_features_per_correlated_block)

        # generate remaining classes without intraclass correlation
        if len(number_of_features_per_correlated_block) < number_of_classes:
            for class_label in range(number_of_classes - len(number_of_features_per_correlated_block)):
                normal_distributed_class = _generate_normal_distributed_class(
                    label=class_label + len(number_of_features_per_correlated_block),
                    number_of_samples=number_of_samples_per_class,
                    number_of_features=number_of_features,
                    scale=scales[class_label + len(number_of_features_per_correlated_block)],
                )
                # generate uncorrelated class feature names
                class_features = []
                for uncorrelated_feature_number in range(normal_distributed_class.shape[1]):
                    class_features.append(["bm_uncorr_" + str(uncorrelated_feature_number)])
                meta_data_dict["class_" + str(class_label)] = class_features

                classes.append(normal_distributed_class)

    # no simulation of intraclass correlation
    else:
        for class_label in range(number_of_classes):
            assert len(scales) == number_of_classes
            normal_distributed_class = _generate_normal_distributed_class(
                label=class_label,
                number_of_samples=number_of_samples_per_class,
                number_of_features=number_of_features,
                scale=scales[class_label],
            )
            # generate uncorrelated class feature names
            class_features = []
            for uncorrelated_feature_number in range(normal_distributed_class.shape[1]):
                class_features.append(["bm_uncorr_" + str(uncorrelated_feature_number)])
            meta_data_dict["class_" + str(class_label)] = class_features
            classes.append(normal_distributed_class)
    assert len(classes) == number_of_classes
    return classes


def _set_default_parameters(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Set default parameters for class shifts.

    Args:
        params_dict: Parameters for data generation (see :func:`generate_artificial_classification_data`).

    Returns:
        Parameters updated with default values for class shifts: each class is shifted by 2 starting with zero.

    """
    # initialize shifts of classes
    if "all_shifts" in params_dict.keys():
        # not all shifts were given correctly by the user: set default shifts
        number_of_classes = (
            params_dict["number_of_normal_distributed_classes"] + params_dict["number_of_lognormal_distributed_classes"]
        )
        all_shifts = []
        for i in range(number_of_classes):
            all_shifts.append(i * 2)
        params_dict["all_shifts"] = all_shifts
    else:
        params_dict["all_shifts"] = (
            params_dict["shifts_of_lognormal_distribution_centers"] + params_dict["means_of_normal_distributions"]
        )

    return params_dict


def _validate_parameters(params_dict) -> bool:
    """Validate given input parameters.

    Args:
        params_dict: Dict including the input parameters to validate (see
            :func:`generate_artificial_classification_data`).

    Returns:
        True, if no error was raised.

    """
    if ("number_of_samples_per_class" not in params_dict.keys()) or (params_dict["number_of_samples_per_class"] <= 0):
        raise ValueError("To generate any data at least one sample must be " 'set for "number_of_samples_per_class".')

    if ("number_of_features_per_class" not in params_dict.keys()) or (params_dict["number_of_features_per_class"] <= 0):
        raise ValueError(
            "In order to generate meaningful classifiable data, "
            "at least one relevant feature per class must be generated.\n "
            "Otherwise, the generated data cannot be used meaningfully "
            "for classification because it is purely random."
        )

    if (
        (params_dict["number_of_normal_distributed_classes"] is None)
        or ("number_of_normal_distributed_classes" not in params_dict.keys())
    ) and (
        (params_dict["number_of_lognormal_distributed_classes"] is None)
        or ("number_of_lognormal_distributed_classes" not in params_dict.keys())
    ):
        raise ValueError("No class is selected to be generated.")

    if (params_dict["number_of_normal_distributed_classes"] <= 0) and (
        params_dict["number_of_lognormal_distributed_classes"] <= 0
    ):
        raise ValueError("The number of classes must be positive.")

    if (
        params_dict["number_of_normal_distributed_classes"] + params_dict["number_of_lognormal_distributed_classes"]
    ) < 2:
        warnings.warn(
            "The number of classes to be generated is is less than two.\n "
            "The generated data cannot be used for classification in this way."
        )

    if (
        not len(params_dict["means_of_normal_distributions"])
        == len(params_dict["scales_of_normal_distributions"])
        == params_dict["number_of_normal_distributed_classes"]
    ):
        warnings.warn(
            "The length of the list of means "
            "(mean_of_normal_distributions) and "
            "scales (scale_of_normal_distributions) for all "
            "normal distributed classes must match the number "
            "of normal distributed classes.\nMean and scale will be set to "
            "default values mean=0 and scale=1."
        )
        params_dict["scales_of_normal_distributions"] = np.ones(params_dict["number_of_normal_distributed_classes"])
        params_dict["means_of_normal_distributions"] = np.zeros(params_dict["number_of_normal_distributed_classes"])

    if params_dict["number_of_samples_per_class"] < 1:
        raise ValueError("Number of samples 'number_of_samples_per_class' must be greater " "than zero. ")

    if (params_dict["means_of_normal_distributions"] is not None) and (
        not params_dict["number_of_normal_distributed_classes"] == len(params_dict["means_of_normal_distributions"])
    ):
        warnings.warn(
            "The number of means given in the "
            '"means_of_normal_distributions" list does not match '
            "the number of normal distributed classes. "
            "All location values are reset to default: Locations of all "
            "classes are shifted by 2 respectively."
        )
        params_dict["all_shifts"] = None

    if params_dict["shifts_of_lognormal_distribution_centers"] is not None:
        if not params_dict["number_of_lognormal_distributed_classes"] == len(
            params_dict["shifts_of_lognormal_distribution_centers"]
        ):
            warnings.warn(
                "The number of shifted locales given in the "
                '"shifts_of_lognormal_distribution_centers" list does not match '
                "the number of lognormal distributed classes.\n "
                "All location values are reset to default: Locations of all "
                "classes are shifted by 2 respectively."
            )
            params_dict["all_shifts"] = None

    if (params_dict["lower_bounds_for_correlations_normal"] is None) and (
        params_dict["number_of_features_per_correlated_block_normal_dist"] is not None
    ):
        warnings.warn(
            '"lower_bounds_for_correlations_normal" is '
            f'{params_dict["lower_bounds_for_correlations_normal"]}'
            f' and "number_of_features_per_correlated_block_normal_dist"'
            f" is "
            f'{params_dict["number_of_features_per_correlated_block_normal_dist"]}.'
            " Lower bounds are all set to default 0.7."
        )
        params_dict["lower_bounds_for_correlations_normal"] = np.full(
            len(params_dict["number_of_features_per_correlated_block_normal_dist"]),
            0.7,
        )

    if (params_dict["upper_bounds_for_correlations_normal"] is None) and (
        params_dict["number_of_features_per_correlated_block_normal_dist"] is not None
    ):
        warnings.warn(
            '"upper_bounds_for_correlations_normal" is '
            f'{params_dict["upper_bounds_for_correlations_normal"]}'
            f' and "number_of_features_per_correlated_block_normal_dist"'
            f" is "
            f'{params_dict["number_of_features_per_correlated_block_normal_dist"]}.'
            " Upper bounds are all set to default 1.0."
        )
        params_dict["upper_bounds_for_correlations_normal"] = np.full(
            len(params_dict["number_of_features_per_correlated_block_normal_dist"]),
            1.0,
        )

    if (params_dict["lower_bounds_for_correlations_lognormal"] is None) and (
        params_dict["number_of_features_per_correlated_block_lognormal"] is not None
    ):
        warnings.warn(
            '"lower_bounds_for_correlations_lognormal" is '
            f'{params_dict["lower_bounds_for_correlations_lognormal"]}'
            f' and "number_of_features_per_correlated_block_lognormal"'
            f" is "
            f'{params_dict["number_of_features_per_correlated_block_lognormal"]}.'
            " Lower bounds are all set to default 0.7."
        )
        params_dict["lower_bounds_for_correlations_lognormal"] = np.full(
            len(params_dict["number_of_features_per_correlated_block_lognormal"]),
            0.7,
        )

    if (params_dict["upper_bounds_for_correlations_lognormal"] is None) and (
        params_dict["number_of_features_per_correlated_block_lognormal"] is not None
    ):
        warnings.warn(
            '"upper_bounds_for_correlations_lognormal" is '
            f'{params_dict["upper_bounds_for_correlations_lognormal"]}'
            f' and "number_of_features_per_correlated_block_lognormal"'
            f" is "
            f'{params_dict["number_of_features_per_correlated_block_lognormal"]}.'
            " Upper bounds are all set to default 1.0."
        )
        params_dict["upper_bounds_for_correlations_lognormal"] = np.full(
            len(params_dict["number_of_features_per_correlated_block_lognormal" ""]),
            1.0,
        )

    if (params_dict["lower_bounds_for_correlations_normal"] is not None) and (
        params_dict["number_of_features_per_correlated_block_normal_dist"] is not None
    ):
        if not (
            len(params_dict["lower_bounds_for_correlations_normal"])
            == len(params_dict["number_of_features_per_correlated_block_normal_dist"])
        ):
            warnings.warn(
                "The number of lower bounds for the correlated features of a "
                "normal distributed class given in the "
                '"lower_bounds_for_correlations_normal" list does not match '
                "the number of blocks with correlating features specified by "
                "the length of the"
                "number_of_features_per_correlated_block_normal_dist list.\n"
                "All lower and upper bound values will be set to default: "
                "lower_bound=0.7 and upper_bound=1.0."
            )
            params_dict["lower_bounds_for_correlations_normal"] = (
                np.full(
                    len(params_dict["number_of_features_per_correlated_block_normal_dist"]),
                    0.7,
                ),
            )

            params_dict["upper_bounds_for_correlations_normal"] = (
                np.full(
                    len(params_dict["number_of_features_per_correlated_block_normal_dist"]),
                    1.0,
                ),
            )

    if (
        (params_dict["upper_bounds_for_correlations_lognormal"] is not None)
        and (params_dict["number_of_features_per_correlated_block_lognormal"] is not None)
        and (
            not len(params_dict["upper_bounds_for_correlations_lognormal"])
            == len(params_dict["number_of_features_per_correlated_block_lognormal"])
        )
    ):
        warnings.warn(
            "The number of upper bounds for the correlated features of a "
            "normal distributed class\ngiven in the "
            '"upper_bounds_for_correlations_lognormal" list does not match '
            "the number of blocks with correlating features specified by "
            "the length of the "
            "number_of_features_per_correlated_block_lognormal list.\n"
            "All lower and upper bound values will be set to default: "
            "lower_bound=0.7 and upper_bound=1.0."
        )
        params_dict["lower_bounds_for_correlations_lognormal"] = (
            np.full(
                len(params_dict["number_of_features_per_correlated_block_lognormal"]),
                0.7,
            ),
        )

        params_dict["upper_bounds_for_correlations_lognormal"] = (
            np.full(
                len(params_dict["number_of_features_per_correlated_block_lognormal"]),
                1.0,
            ),
        )

    if (params_dict["upper_bounds_for_correlations_normal"] is not None) and not len(
        params_dict["upper_bounds_for_correlations_normal"]
    ) == len(params_dict["number_of_features_per_correlated_block_normal_dist"]):
        warnings.warn(
            "The number of upper bounds for the correlated features of a "
            "normal distributed class given in the "
            '"upper_bounds_for_correlations_normal" list does not match '
            "the number of blocks with correlating features specified by "
            "the length of the"
            "number_of_features_per_correlated_block_normal_dist list.\n"
            "All lower and upper bound values will be set to default: "
            "lower_bound=0.7 and upper_bound=1.0."
        )
        params_dict["lower_bounds_for_correlations_normal"] = (
            np.full(
                len(params_dict["number_of_features_per_correlated_block_normal_dist"]),
                0.7,
            ),
        )

        params_dict["upper_bounds_for_correlations_normal"] = (
            np.full(
                len(params_dict["number_of_features_per_correlated_block_normal_dist"]),
                1.0,
            ),
        )

    if (params_dict["upper_bounds_for_correlations_lognormal"] is not None) and (
        params_dict["number_of_features_per_correlated_block_lognormal"] is not None
    ):
        if not (
            len(params_dict["upper_bounds_for_correlations_lognormal"])
            == len(params_dict["number_of_features_per_correlated_block_lognormal"])
        ):
            warnings.warn(
                "The number of upper bounds for the correlated features of a "
                "normal distributed class given in the "
                '"upper_bounds_for_correlations_lognormal" list does not '
                "match "
                "the number of blocks with correlating features specified by "
                "the length of the"
                "number_of_features_per_correlated_block_lognormal list.\n"
                "All lower and upper bound values will be set to default: "
                "lower_bound=0.7 and upper_bound=1.0."
            )
            params_dict["lower_bounds_for_correlations_lognormal"] = (
                np.full(
                    len(params_dict["number_of_features_per_correlated_block_lognormal"]),
                    0.7,
                ),
            )

            params_dict["upper_bounds_for_correlations_lognormal"] = (
                np.full(
                    len(params_dict["number_of_features_per_correlated_block_lognormal"]),
                    1.0,
                ),
            )

    return True


def generate_artificial_data(params_dict: Dict[str, Any]):
    """Generate artificial biomarker data.

    Args:
        params_dict: Parameters for the data to generate (see
            parameters of :func:`generate_artificial_classification_data`).

    Returns:
        Generated artificial data as DataFrame.
    """
    # validate input parameters
    _validate_parameters(params_dict)
    params_dict = _set_default_parameters(params_dict)

    total_number_of_classes = (
        params_dict["number_of_normal_distributed_classes"] + params_dict["number_of_lognormal_distributed_classes"]
    )
    number_of_all_samples = params_dict["number_of_samples_per_class"] * total_number_of_classes

    meta_data_dict: dict = {}

    # generate labels
    labels = np.asarray(range(total_number_of_classes), dtype=np.float64)

    number_of_lognormal_distributed_classes = params_dict["number_of_lognormal_distributed_classes"]

    # generate lognormal distributed classes
    lognormal_distributed_classes_list = _generate_normal_distributed_classes(
        labels[: params_dict["number_of_lognormal_distributed_classes"]],
        meta_data_dict,
        params_dict["number_of_features_per_class"],
        params_dict["number_of_lognormal_distributed_classes"],
        params_dict["number_of_features_per_correlated_block_lognormal"],
        params_dict["number_of_samples_per_class"],
        params_dict["scales_of_normal_distributions"],
        params_dict["lower_bounds_for_correlations_lognormal"],
        params_dict["upper_bounds_for_correlations_lognormal"],
    )

    # transform normal distributions to lognormal distributions to simulate
    # the samples from ill patients including extreme values and outliers
    lognormal_distributed_classes_list = _transform_normal_to_lognormal(lognormal_distributed_classes_list)

    assert params_dict["number_of_lognormal_distributed_classes"] == len(lognormal_distributed_classes_list)
    # label of first element of first class should be zero
    assert lognormal_distributed_classes_list[0][0, 0] == labels[0]

    # generate normal distributed classes
    normal_distributed_classes_list = _generate_normal_distributed_classes(
        labels[number_of_lognormal_distributed_classes:],
        meta_data_dict,
        params_dict["number_of_features_per_class"],
        params_dict["number_of_normal_distributed_classes"],
        params_dict["number_of_features_per_correlated_block_normal_dist"],
        params_dict["number_of_samples_per_class"],
        params_dict["scales_of_normal_distributions"],
        params_dict["lower_bounds_for_correlations_normal"],
        params_dict["upper_bounds_for_correlations_normal"],
    )
    assert params_dict["number_of_normal_distributed_classes"] == len(normal_distributed_classes_list)

    artificial_classes_list = lognormal_distributed_classes_list + normal_distributed_classes_list

    # shift all classes
    artificial_classes_list, classes_df = _shift_all_classes(artificial_classes_list, params_dict)

    assert total_number_of_classes == len(artificial_classes_list)

    # visualize distributions
    _visualize_distributions(classes_df)

    complete_classes = np.concatenate(artificial_classes_list, axis=0)

    # check the data shape and the correct generation of the labels
    print(complete_classes.shape)
    assert complete_classes.shape[0] == params_dict["number_of_samples_per_class"] * total_number_of_classes
    assert (
        complete_classes.shape[1]
        == params_dict["number_of_features_per_class"] + 1
        # for the label
    )
    # cast labels to list of int
    for class_label in np.int_(labels).tolist():
        assert complete_classes[params_dict["number_of_samples_per_class"] * class_label, 0] == class_label

    # append pseudo class
    pseudo_class = _generate_pseudo_class(params_dict)
    complete_classes = np.concatenate((complete_classes, pseudo_class), axis=1)
    assert (
        complete_classes.shape[1]
        == params_dict["number_of_features_per_class"]
        + params_dict["number_of_pseudo_class_features"]
        + 1  # for the label
    )

    # append random features
    random_features = np.random.normal(
        loc=0.0,
        scale=2,
        size=(number_of_all_samples, params_dict["number_of_random_features"]),
    )
    complete_data_set = np.concatenate((complete_classes, random_features), axis=1)

    # check final data shape
    print(complete_data_set.shape)
    assert complete_data_set.shape[0] == params_dict["number_of_samples_per_class"] * total_number_of_classes
    assert (
        complete_data_set.shape[1]
        == params_dict["number_of_features_per_class"]
        + params_dict["number_of_pseudo_class_features"]
        + 1  # for the label
        + params_dict["number_of_random_features"]
    )

    complete_data_df = pd.DataFrame(complete_data_set)

    # generate feature names
    _generate_column_names(complete_data_df, params_dict)

    return complete_data_df, meta_data_dict


def save_result(
    data_df: pd.DataFrame,
    path_to_save_csv=None,
    path_to_save_feather=None,
) -> None:
    """Save the generated data.

    Args:
        data_df: DataFrame to be saved.
        path_to_save_csv: Path for saving the generated data as csv. Default is None.
        path_to_save_feather: Path for saving the generated data as feather. Default is None.

    """
    if path_to_save_csv is not None:
        assert isinstance(path_to_save_csv, str)
        pd.DataFrame(data_df).to_csv(path_to_save_csv, index=False)

        print(f"Data generated successfully and saved in " f"{path_to_save_csv}")

    if path_to_save_feather is not None:
        assert isinstance(path_to_save_feather, str)
        pd.DataFrame(data_df).to_feather(path_to_save_feather, index=False)

        print(f"Data generated successfully and saved in " f"{path_to_save_feather}")


def save_meta_data(
    meta_data: Dict[str, list],
    path_to_save_meta_data=None,
) -> None:
    """Save the generated data.

    Args:
        path_to_save_meta_data: Path for saving the metadata. Default is None.
        meta_data: Dict including metadata to be saved.

    """
    if path_to_save_meta_data is not None:
        assert isinstance(path_to_save_meta_data, str)
        joblib.dump(meta_data, path_to_save_meta_data)

        print(f"Meta data successfully saved in " f"{path_to_save_meta_data}")


def generate_artificial_classification_data(
    number_of_samples_per_class: int,
    number_of_features_per_class: int,
    number_of_normal_distributed_classes: int = 0,
    number_of_lognormal_distributed_classes: int = 0,
    means_of_normal_distributions: Union[List[Number], ndarray] = None,
    scales_of_normal_distributions: Union[List[Number], ndarray] = None,
    scales_of_lognormal_distributions: Union[List[Number], ndarray] = None,
    shifts_of_lognormal_distribution_centers: Union[List[Number], ndarray] = None,
    number_of_features_per_correlated_block_normal_dist: Optional[Union[List[List[int]], ndarray]] = None,
    lower_bounds_for_correlations_normal: Optional[Union[List[Number], ndarray]] = None,
    upper_bounds_for_correlations_normal: Optional[Union[List[Number], ndarray]] = None,
    number_of_features_per_correlated_block_lognormal: Optional[Union[List[List[int]], ndarray]] = None,
    lower_bounds_for_correlations_lognormal: Optional[Union[List[Number], ndarray]] = None,
    upper_bounds_for_correlations_lognormal: Optional[Union[List[Number], ndarray]] = None,
    number_of_pseudo_class_features: int = 0,
    number_of_random_features: int = 0,
    path_to_save_plot: str = None,
    path_to_save_csv: str = None,
    path_to_save_feather: str = None,
    path_to_save_meta_data: str = None,
    shuffle_features: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate artificial classification (e.g. biomarker) data.

    Args:
        number_of_normal_distributed_classes: Number of classes with a normal distribution.
        means_of_normal_distributions: Means of normal distributed classes.
            Default is zero and a shift of 2 for each class.
        scales_of_normal_distributions: Scales of normal distributed classes. Default is scale 1 for each class.
        scales_of_lognormal_distributions: Scales of the underlying normal distributions of the
            lognormal distributed classes. Default is scale 1 for each underlying normal distribution.
        number_of_lognormal_distributed_classes: Number of classes with a lognormal distribution
            to simulate outliers and extreme values.
        shifts_of_lognormal_distribution_centers: Shifts for all lognormal distributed classes.
            Default is a shift of 2 for each class respectively.
        number_of_samples_per_class: Number of samples (rows) to generate for each artificial class.
        number_of_features_per_class: Total number of features (columns) to generate for each artificial class.
        number_of_features_per_correlated_block_normal_dist: Number of features (columns) to generate for each block
        of correlated features within a normal distributed class for the simulation of intra class correlation.
        lower_bounds_for_correlations_normal: Lower bounds for the correlation of each block of correlated features
            within normal distributed class. Default is 0.7.
        upper_bounds_for_correlations_normal: Upper bounds for the correlation of each block of correlated features
            within a normal distributed class. Default is 1.0.
        number_of_features_per_correlated_block_lognormal: Number of features (columns) for each block of correlated
            features within a lognormal distributed class. This can simulate intra-class correlation.
            In biology, for example, this can be the activation of complete pathways or redundant systems.
        lower_bounds_for_correlations_lognormal: Lower bounds for the correlation of each block of correlated features
            within a lognormal distributed class. Default is 0.7.
        upper_bounds_for_correlations_lognormal:Upper bounds for the correlation of each block of correlated features
            within a lognormal distributed class. Default is 1.0.
        number_of_pseudo_class_features: Number of pseudo-class features.
            The underlying classes correspond to the selected classes for lognormal and normal distributions and
            the given shifts of the generated artificial classes or their default values. All samples of the generated
            classes are randomly shuffled and therefore have no relation to any class label.
        number_of_random_features: Number of randomly generated features.
            For their generation, a normal distribution with mean zero and scale 2 is used.
        path_to_save_plot: Path to save generated plots.
        path_to_save_csv: Path to save generated data as csv.
        path_to_save_feather: Path to save generated data in the feather format.
        path_to_save_meta_data: Path to pickle metadata (names of correlated features) of the generated blocks
            of correlated features.
        shuffle_features: Generate artificial classification data with shuffled features.

    Returns:
        Generated artificial data and metadata for correlated features.
    """
    # Save variables to adjust the line length in dict
    num_features_corr_lognormal = number_of_features_per_correlated_block_normal_dist
    num_features_corr_normal = number_of_features_per_correlated_block_lognormal

    params_dict = dict(
        number_of_normal_distributed_classes=number_of_normal_distributed_classes,
        means_of_normal_distributions=means_of_normal_distributions,
        scales_of_normal_distributions=scales_of_normal_distributions,
        scales_of_lognormal_distributions=scales_of_lognormal_distributions,
        number_of_lognormal_distributed_classes=number_of_lognormal_distributed_classes,
        shifts_of_lognormal_distribution_centers=shifts_of_lognormal_distribution_centers,
        number_of_samples_per_class=number_of_samples_per_class,
        number_of_features_per_class=number_of_features_per_class,
        number_of_features_per_correlated_block_normal_dist=num_features_corr_normal,
        lower_bounds_for_correlations_normal=lower_bounds_for_correlations_normal,
        upper_bounds_for_correlations_normal=upper_bounds_for_correlations_normal,
        number_of_features_per_correlated_block_lognormal=num_features_corr_lognormal,
        lower_bounds_for_correlations_lognormal=lower_bounds_for_correlations_lognormal,
        upper_bounds_for_correlations_lognormal=upper_bounds_for_correlations_lognormal,
        number_of_pseudo_class_features=number_of_pseudo_class_features,
        number_of_random_features=number_of_random_features,
        path_to_save_plot=path_to_save_plot,
        path_to_save_csv=path_to_save_csv,
        path_to_save_feather=path_to_save_feather,
    )
    complete_data_df, meta_data_dict = generate_artificial_data(params_dict)

    if shuffle_features:
        # shuffle artificial features
        column_names = list(complete_data_df.columns[1:])
        random.shuffle(column_names)
        shuffled_column_names = ["label"] + column_names
        complete_data_df = complete_data_df[shuffled_column_names]

        assert complete_data_df.columns[0] == "label"

    save_result(complete_data_df, path_to_save_csv, path_to_save_feather)
    save_meta_data(meta_data_dict, path_to_save_meta_data)

    return complete_data_df, meta_data_dict

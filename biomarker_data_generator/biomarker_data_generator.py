# Copyright (c) 2022 Sigrun May,
# Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
"""
Generator for artificial biomarker data from high throughput experiments.

Can be used as baseline for benchmarking and the development of new methods.
"""

import math
import random
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from matplotlib import pyplot
from numpy.random import default_rng
from sklearn.utils import shuffle
from statsmodels.stats import correlation_tools


def generate_normal_distributed_class(
    label: int,
    number_of_samples: int,
    number_of_biomarkers: int,
    scale,
):
    """
    Generate artificial data to simulate the samples from healthy patients.

    :param label: Label for the generated artificial class.
    :param number_of_samples: Number of rows of generated data.
    :param number_of_biomarkers: Number of columns of generated data.
    :param mean_of_normal_distribution: Mean of the data distribution.
    :param scale:
    :return: Normal distributed data of the given shape and parameters
    with the given label in the first column.
    """
    # generate labels
    label_vector = np.full((number_of_samples, 1), label)

    # generate data
    rng = default_rng()
    features = rng.normal(
        loc=0,
        scale=scale,
        size=(number_of_samples, number_of_biomarkers),
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
        )

    normal_distributed_data = np.hstack((label_vector, features))
    return normal_distributed_data

    # Generate artificial data to simulate the samples from
    # ill patients including extreme values and outliers.
    #
    # :param label: Label for the generated artificial class.
    # :param number_of_samples: Number of rows of generated data.
    # :param number_of_biomarkers: Number of columns of generated data.
    # :param shift_of_lognormal_distribution: Shift of the
    # lognormal distribution simulate different classes.
    # :param mean_lognormal_distribution: Mean value of the underlying
    # normal distribution. Default is zero.
    # :param sigma_of_lognormal_distribution: Standard deviation of the
    # underlying normal distribution.
    # Should be greater than zero. Default is 1.
    # :return: Lognormal distributed data of the given shape and
    # parameters with the given label in the first column.


def generate_pseudo_class(params_dict: dict):
    """Create a pseudo-class by shuffling artificial classes.

    Creates a pseudo-class by shuffling the specified number of artificial
    classes. The total number of classes should match the total number of
    classes for other data blocks. This allows different data blocks to be
    merged seamlessly and the pseudo-classes to match the number of real
    labels. To simulate different classes, the lognormal distributions are
    shifted by steps of width step_width_shift_lognormal_distributed_class
    starting with step_width_shift_lognormal_distributed_class.
    The means of normal distributed classes are shifted by steps of width
    step_width_shift_normal_distributed_class starting with 0. Using the
    default values, the artificial classes can be clearly distinguished
    from each other.

    :param params_dict:
    # :param number_of_pseudo_class_features: Number of columns of generated
    # data.
    # :param number_of_normal_distributed_classes: Number of normal
    # distributed classes.
    # :param number_of_lognormal_distributed_classes: Number of lognormal
    # distributed classes.
    # :param number_of_samples_per_class: Number of rows of generated data
    # for each class.
    # :param step_width_shift_lognormal_distributed_class: Step size for
    # shifting a lognormal distribution
    # in order to clearly distinguish artificial classes from each other.
    # Default is 6.
    # :param step_width_shift_mean_normal_distributed_class: Step size for
    # shifting the mean of a normal distribution
    # in order to clearly distinguish artificial classes from each other.
    # Default is 10.
    :return: Randomly shuffled pseudo-class with the given number of specific
    classes: Numpy array of the given shape.
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


def generate_normal_distributed_correlated_block(
    number_of_features,
    number_of_samples,
    # mean_list,
    lower_bound,
    upper_bound,
):
    """Generate a block of correlated features.

    :param label: Label for the generated artificial class.
    :param number_of_features: Number of columns of generated data.
    :param number_of_samples: Number of rows of generated data.
    # :param mean_list:
    :param lower_bound:
    :param upper_bound:
    :return: Numpy array of the given shape with correlating features
    in the given range.
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
    correlation_matrix = correlation_tools.corr_nearest(
        corr=random_matrix, threshold=1e-15, n_fact=100
    )

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
    # _visualize_correlations(covariant_class)

    return covariant_block


def _visualize_correlations(data):
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

    sns.heatmap(corr, cmap="YlGnBu")
    pyplot.show()


def _visualize_distributions(data):
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
    sns.displot(data=data_df, kind="kde")
    pyplot.show()


def _generate_column_names(
    data_df,
    params_dict,
):
    """Generate semantic names for the columns of the given DataFrame.

    Args:
        data_df: DataFrame with unnamed columns.
        params_dict:

    Returns: DataFrame with meaningful named columns:
                        "bm" for artificial biomarker
                        "pseudo" for pseudo-class feature
                        "random" for random data

    """
    # :param meta_data_dict:
    # :param number_of_biomarkers: Number of artificial
    # biomarkers inculded in the given DataFrame.
    # :param number_of_pseudo_class_features: Number of
    # pseudo-class features inculded in the given DataFrame.
    # :param number_of_random_features: Number of random
    # features inculded in the given DataFrame.
    # :return: DataFrame with meaningful named columns:
    # bm for artificial biomarker
    # pseudo for pseudo-class feature
    # random for random data

    # generate label as first entry
    column_names = ["label"]

    # generate names for artificial biomarkers
    for column_name in range(params_dict["number_of_artificial_biomarkers"]):
        column_names.append("bm_" + str(column_name))

    for column_name in range(params_dict["number_of_pseudo_class_features"]):
        column_names.append("pseudo_" + str(column_name))

    for column_name in range(params_dict["number_of_random_features"]):
        column_names.append("random_" + str(column_name))

    data_df.columns = column_names
    return data_df


def _shift_all_classes(classes_list: list, params_dict: dict):
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


def _transform_normal_to_lognormal(classes_list: list):
    """Transform normal distributed data to lognormal distributions.

    Args:
        classes_list: List of classes as numpy arrays.

    Returns: List of classes as numpy arrays transformed to lognormal
    distributions.

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
    labels,
    meta_data_dict,
    number_of_features,
    number_of_classes,
    number_of_features_per_correlated_block,
    number_of_samples_per_class,
    scales,
    lower_bounds_for_correlations,
    upper_bounds_for_correlations,
) -> list:
    """Generate artificial classes with the given parameters as numpy arrays.

    Args:
        labels:
        meta_data_dict:
        number_of_features:
        number_of_classes:
        number_of_features_per_correlated_block:
        number_of_samples_per_class:
        scales:
        lower_bounds_for_correlations:
        upper_bounds_for_correlations:

    Returns: List of generated classes of the given shape as numpy arrays.

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
                generate_normal_distributed_correlated_block(
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
            meta_data_dict["class_" + str(label) + "_blocks"] = len(
                number_of_features_per_correlated_block[i]
            )
            meta_data_dict["class_" + str(label)] = class_features

            generated_class = np.concatenate(blocks, axis=1)

            # _visualize_correlations(generated_class)

            # generate uncorrelated features
            if generated_class.shape[1] < number_of_features:
                uncorrelated_features = generate_normal_distributed_class(
                    label,
                    number_of_samples_per_class,
                    number_of_biomarkers=number_of_features - generated_class.shape[1],
                    scale=scales[i],
                )
                unlabeled_uncorrelated_features = uncorrelated_features[:, 1:]
                # exclude the label as it is appended later
                generated_class = np.concatenate(
                    (generated_class, unlabeled_uncorrelated_features), axis=1
                )
                # generate unlabeled uncorrelated class feature names
                for uncorrelated_feature_number in range(
                    unlabeled_uncorrelated_features.shape[1]
                ):
                    class_features.append(["uncorr_" + str(uncorrelated_feature_number)])
                meta_data_dict["class_" + str(label)] = class_features

            assert generated_class.shape[1] == number_of_features
            assert generated_class.shape[0] == number_of_samples_per_class

            # generate names to check the number of generated features
            assert isinstance(meta_data_dict, dict)
            class_feature_names_array = np.concatenate(
                [np.array(i) for i in meta_data_dict["class_" + str(label)]]
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
            for class_label in range(
                number_of_classes - len(number_of_features_per_correlated_block)
            ):
                normal_distributed_class = generate_normal_distributed_class(
                    label=class_label + len(number_of_features_per_correlated_block),
                    number_of_samples=number_of_samples_per_class,
                    number_of_biomarkers=number_of_features,
                    scale=scales[
                        class_label + len(number_of_features_per_correlated_block)
                    ],
                )
                # generate uncorrelated class feature names
                class_features = []
                for uncorrelated_feature_number in range(
                    normal_distributed_class.shape[1]
                ):
                    class_features.append(
                        ["bm_uncorr_" + str(uncorrelated_feature_number)]
                    )
                meta_data_dict["class_" + str(class_label)] = class_features

                classes.append(normal_distributed_class)

    # no simulation of intraclass correlation
    else:
        for class_label in range(number_of_classes):
            assert len(scales) == number_of_classes
            normal_distributed_class = generate_normal_distributed_class(
                label=class_label,
                number_of_samples=number_of_samples_per_class,
                number_of_biomarkers=number_of_features,
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


def _set_parameters(params_dict):
    # initialize shifts of classes
    if "all_shifts" in params_dict.keys():
        # not all shifts were given correctly by the user: set default shifts
        number_of_classes = (
            params_dict["number_of_normal_distributed_classes"]
            + params_dict["number_of_lognormal_distributed_classes"]
        )
        all_shifts = []
        for i in range(number_of_classes):
            all_shifts.append(i * 2)
        params_dict["all_shifts"] = all_shifts
    else:
        params_dict["all_shifts"] = (
            params_dict["shifts_of_lognormal_distribution_centers"]
            + params_dict["means_of_normal_distributions"]
        )

    return params_dict


def _validate_parameters(params_dict):
    if (
        not len(params_dict["means_of_normal_distributions"])
        == len(params_dict["scales_of_normal_distributions"])
        == params_dict["number_of_normal_distributed_classes"]
    ):
        raise ValueError(
            "The length of the list of means "
            "(mean_of_normal_distributions) and "
            "scales (scale_of_normal_distributions) for all "
            "normal distributed classes must match the number "
            "of normal distributed classes."
        )
    if not (
        len(params_dict["shifts_of_lognormal_distribution_centers"])
        == params_dict["number_of_lognormal_distributed_classes"]
    ):
        raise ValueError(
            "The length of the list of shifts "
            "(shifts_of_lognormal_distribution_centers) for "
            "all lognormal distributed classes "
            "must match the number of lognormal distributed classes."
        )
    if (
        params_dict["number_of_normal_distributed_classes"]
        + params_dict["number_of_lognormal_distributed_classes"]
    ) < 2:
        warnings.warn(
            "The number of classes to be generated is is less than two. "
            "The generated data cannot be used for classification in this way."
        )

    if params_dict["number_of_samples_per_class"] < 1:
        raise ValueError(
            "Number of samples number_of_samples_per_class " "must be greater than zero."
        )

    if not params_dict["number_of_normal_distributed_classes"] == len(
        params_dict["means_of_normal_distributions"]
    ):
        warnings.warn(
            "No or not all mean values are given in the "
            '"means_of_normal_distributions" list. All mean '
            "values are reset to default: Means of all classes are "
            "shifted by 2 each."
        )
        params_dict["all_shifts"] = None

    if not params_dict["number_of_lognormal_distributed_classes"] == len(
        params_dict["shifts_of_lognormal_distribution_centers"]
    ):
        warnings.warn(
            "No or not all mean values are given in the "
            '"shifts_of_lognormal_distribution_centers" list. '
            "All mean values are reset to default: Means of all "
            "classes are shifted by 2 each."
        )
        params_dict["all_shifts"] = None

    return True


def generate_artificial_data(params_dict: dict):
    """Generate artificial biomarker data.

    Args:
        params_dict:

    Returns: Generated artificial data as DataFrame.

    """
    params_dict = _set_parameters(params_dict)
    # TODO: update to different sample numbers per class
    total_number_of_classes = (
        params_dict["number_of_normal_distributed_classes"]
        + params_dict["number_of_lognormal_distributed_classes"]
    )
    number_of_all_samples = (
        params_dict["number_of_samples_per_class"] * total_number_of_classes
    )

    meta_data_dict: Dict[str, list] = {}

    # generate labels
    labels = list(range(total_number_of_classes))

    number_of_lognormal_distributed_classes = params_dict[
        "number_of_lognormal_distributed_classes"
    ]

    # generate lognormal distributed classes
    lognormal_distributed_classes_list = _generate_normal_distributed_classes(
        labels[: params_dict["number_of_lognormal_distributed_classes"]],
        meta_data_dict,
        params_dict["number_of_artificial_biomarkers"],
        params_dict["number_of_lognormal_distributed_classes"],
        params_dict["number_of_features_per_correlated_block"],
        # params_dict["number_of_intra_class_correlated_blocks_lognormal"],
        params_dict["number_of_samples_per_class"],
        params_dict["scales_of_normal_distributions"],
        params_dict["lower_bounds_for_correlations_lognormal"],
        params_dict["upper_bounds_for_correlations_lognormal"],
    )

    # transform normal distributions to lognormal distributions to simulate
    # the samples from ill patients including extreme values and outliers
    lognormal_distributed_classes_list = _transform_normal_to_lognormal(
        lognormal_distributed_classes_list
    )

    assert params_dict["number_of_lognormal_distributed_classes"] == len(
        lognormal_distributed_classes_list
    )
    # label of first element of first class should be zero
    assert lognormal_distributed_classes_list[0][0, 0] == labels[0]

    # generate normal distributed classes
    normal_distributed_classes_list = _generate_normal_distributed_classes(
        labels[number_of_lognormal_distributed_classes:],
        meta_data_dict,
        params_dict["number_of_artificial_biomarkers"],
        params_dict["number_of_normal_distributed_classes"],
        params_dict["number_of_features_per_correlated_block"],
        # params_dict["number_of_intra_class_correlated_blocks_normal"],
        params_dict["number_of_samples_per_class"],
        params_dict["scales_of_normal_distributions"],
        params_dict["lower_bounds_for_correlations_normal"],
        params_dict["upper_bounds_for_correlations_normal"],
    )
    assert params_dict["number_of_normal_distributed_classes"] == len(
        normal_distributed_classes_list
    )

    artificial_classes_list = (
        lognormal_distributed_classes_list + normal_distributed_classes_list
    )

    # shift all classes
    artificial_classes_list, classes_df = _shift_all_classes(
        artificial_classes_list, params_dict
    )

    assert total_number_of_classes == len(artificial_classes_list)

    # visualize correlations
    _visualize_correlations(classes_df)

    complete_classes = np.concatenate(artificial_classes_list, axis=0)

    # check the data shape and the correct generation of the labels
    print(complete_classes.shape)
    assert (
        complete_classes.shape[0]
        == params_dict["number_of_samples_per_class"] * total_number_of_classes
    )
    assert (
        complete_classes.shape[1]
        == params_dict["number_of_artificial_biomarkers"] + 1
        # for the label
    )
    for class_label in labels:
        assert (
            complete_classes[params_dict["number_of_samples_per_class"] * class_label, 0]
            == class_label
        )

    # append pseudo class
    pseudo_class = generate_pseudo_class(params_dict)
    complete_classes = np.concatenate((complete_classes, pseudo_class), axis=1)
    assert (
        complete_classes.shape[1]
        == params_dict["number_of_artificial_biomarkers"]
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
    assert (
        complete_data_set.shape[0]
        == params_dict["number_of_samples_per_class"] * total_number_of_classes
    )
    assert (
        complete_data_set.shape[1]
        == params_dict["number_of_artificial_biomarkers"]
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
):
    """Save the generated data.

    Args:
        path_to_save_csv: Path for saving the generated data as csv.
        Default is None.
        path_to_save_feather: Path for saving the generated data as feather.
        Default is None.
        data_df: DataFrame to be saved.

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
):
    """Save the generated data.

    Args:
        path_to_save_meta_data: Path for saving the meta data.
        Default is None.
        meta_data: Dict including meta data to be saved.

    """
    if path_to_save_meta_data is not None:
        assert isinstance(path_to_save_meta_data, str)
        joblib.dump(meta_data, "../data/meta_data_complete_artif.pkl")

        print(f"Meta data successfully saved in " 
              f"{path_to_save_meta_data}")


def generate_shuffled_artificial_data(params_dict: dict):
    """Generate artificial biomarker data with shuffled features.

    Args:
        params_dict:

    Returns: Generated artificial data with shuffled features as DataFrame.

    """
    complete_data_df, meta_data_dict = generate_artificial_data(params_dict)

    # shuffle artificial features
    column_names = list(complete_data_df.columns[1:])
    random.shuffle(column_names)
    shuffled_column_names = ["label"] + column_names
    shuffled_data_df = complete_data_df[shuffled_column_names]

    assert shuffled_data_df.columns[0] == "label"

    return shuffled_data_df, meta_data_dict
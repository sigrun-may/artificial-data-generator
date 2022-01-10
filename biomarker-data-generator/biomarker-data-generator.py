"""

"""

import math
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.utils import shuffle
from sklearn.datasets import make_spd_matrix
import seaborn as sns
from matplotlib import pyplot
from statsmodels.stats import correlation_tools
import warnings
import random

# Settings
PATH_TO_SAVE_CSV = "../data/artifical_biological_data.csv"
PATH_TO_SAVE_FEATHER_DATA = None  # if None, generated data will not be saved

NUMBER_OF_BIOMARKERS = 6
NUMBER_OF_PSEUDO_CLASS_FEATURES = 10
NUMBER_OF_RANDOM_FEATURES = 1000

# each class is assigned the same number of samples TODO: different sample numbers
# total number of samples = number of classes * number of samples per class
NUMBER_OF_SAMPLES_PER_CLASS = 10

# scale = standard deviation of normal distribution
SCALE = 1
# sigma = standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
SIGMA = 1
MEAN_NORMAL_DISTRIBUTION = 1
MEAN_LOGNORMAL_DISTRIBUTION = 0
SHIFT_LOGNORMAL_DISTRIBUTION = 10
NUMBER_OF_LOGNORMAL_DISTRIBUTED_CLASSES = 1
NUMBER_OF_NORMAL_DISTRIBUTED_CLASSES = 1


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
    :return: Normal distributed data of the given shape and parameters with the given label in the first column.
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
            + " -> Try choosing a smaller scale for a small sample size or accept a deviating mean."
        )

    normal_distributed_data = np.hstack((label_vector, features))
    return normal_distributed_data


def generate_lognormal_distributed_class(
    label: int,
    number_of_samples: int,
    number_of_biomarkers: int,
    shift_of_lognormal_distribution: float,
    mean_lognormal_distribution=0,
    sigma_of_lognormal_distribution=1,
):
    """
    Generate artificial data to simulate the samples from ill patients including extreme values and outliers.
    :param label: Label for the generated artificial class.
    :param number_of_samples: Number of rows of generated data.
    :param number_of_biomarkers: Number of columns of generated data.
    :param shift_of_lognormal_distribution: Shift of the lognormal distribution simulate different classes.
    :param mean_lognormal_distribution: Mean value of the underlying normal distribution. Default is zero.
    :param sigma_of_lognormal_distribution: Standard deviation of the underlying normal distribution.
    Should be greater than zero. Default is 1.
    :return: Lognormal distributed data of the given shape and parameters with the given label in the first column.
    """
    # generate labels
    label_vector = np.full((number_of_samples, 1), label)

    # generate data
    rng = default_rng()
    features = rng.lognormal(
        mean=mean_lognormal_distribution,
        sigma=sigma_of_lognormal_distribution,
        size=(number_of_samples, number_of_biomarkers),
    )
    features = features + shift_of_lognormal_distribution
    lognormal_distributed_data = np.hstack((label_vector, features))
    return lognormal_distributed_data

    # data_df = pd.read_csv(
    #     "../data/6bm_lognormal_correlated.csv", sep=";", header=None, index_col=None
    # )
    # data_array = data_df.values
    # shifted_and_scaled_data = (data_array[:, 1:] / 20) + 3.5
    # complete_data = np.hstack(
    #     (data_array[:, 0].reshape(-1, 1), shifted_and_scaled_data)
    # )
    # return complete_data


def generate_pseudo_class(
    number_of_pseudo_class_features,
    number_of_normal_distributed_classes,
    number_of_lognormal_distributed_classes,
    number_of_samples_per_class,
    step_width_shift_lognormal_distributed_class=6,
    step_width_shift_mean_normal_distributed_class=10,
):
    """
    Creating a pseudo-class by shuffling the specified number of artificial classes.
    The total number of classes should match the total number of classes for other
    data blocks. This allows different data blocks to be merged seamlessly and
    the pseudo-classes to match the number of real labels.
    To simulate different classes, the lognormal distributions are shifted
    by steps of width step_width_shift_lognormal_distributed_class starting with
    step_width_shift_lognormal_distributed_class.
    The means of normal distributed classes are shifted by steps of width
    step_width_shift_normal_distributed_class starting with 0.
    Using the default values, the artificial classes can be clearly distinguished
    from each other.

    :param number_of_pseudo_class_features: Number of columns of generated data.
    :param number_of_normal_distributed_classes: Number of normal distributed classes.
    :param number_of_lognormal_distributed_classes: Number of lognormal distributed classes.
    :param number_of_samples_per_class: Number of rows of generated data for each class.
    :param step_width_shift_lognormal_distributed_class: Step size for shifting a lognormal distribution
    in order to clearly distinguish artificial classes from each other. Default is 6.
    :param step_width_shift_mean_normal_distributed_class: Step size for shifting the mean of a normal distribution
    in order to clearly distinguish artificial classes from each other. Default is 10.
    :return: Randomly shuffled pseudo-class with the given number of specific classes: Numpy array of the given shape.
    """
    rng = default_rng()

    # generate first part of the pseudo class
    simulated_classes = (
        rng.lognormal(
            size=(number_of_samples_per_class, number_of_pseudo_class_features)
        )
        + step_width_shift_lognormal_distributed_class
    )

    # generate further lognormal distributed classes
    for i in range(1, number_of_lognormal_distributed_classes):
        assert number_of_lognormal_distributed_classes > 1

        # shift random lognormal data to generate different classes
        shifted_lognormal_distributed_class = rng.lognormal(
            size=(number_of_samples_per_class, number_of_pseudo_class_features)
        ) + (step_width_shift_lognormal_distributed_class * (i + 1))
        simulated_classes = np.vstack(
            (simulated_classes, shifted_lognormal_distributed_class)
        )

    # generate normal distributed classes
    for j in range(number_of_normal_distributed_classes):
        # shift random data to generate different classes
        shifted_standard_normal_class = rng.standard_normal(
            size=(number_of_samples_per_class, number_of_pseudo_class_features)
        ) + (step_width_shift_mean_normal_distributed_class * j)

        if not math.isclose(
            np.mean(shifted_standard_normal_class),
            (step_width_shift_mean_normal_distributed_class * j),
            abs_tol=0.4,
        ):
            print(
                f"INFO: Mean {np.mean(shifted_standard_normal_class)} of generated data "
                f"differs from expected mean {step_width_shift_mean_normal_distributed_class * j} within the pseudo class."
            )

        simulated_classes = np.vstack(
            (simulated_classes, shifted_standard_normal_class)
        )

    assert simulated_classes.shape == (
        (
            (
                number_of_lognormal_distributed_classes
                + number_of_normal_distributed_classes
            )
            * number_of_samples_per_class
        ),
        number_of_pseudo_class_features,
    )

    # shuffle classes to finally create pseudo-class
    pseudo_class = shuffle(pd.DataFrame(simulated_classes))
    return np.array(pseudo_class)


def generate_normal_distributed_correlated_block(
    number_of_features,
    number_of_samples,
    # mean_list,
    lower_bound,
    upper_bound,
):
    """

    :param label: Label for the generated artificial class.
    :param number_of_features: Number of columns of generated data.
    :param number_of_samples: Number of rows of generated data.
    # :param mean_list:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    rng = default_rng()

    # generate random matrix to constrain a range of values and specify a starting point
    random_matrix = np.random.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(number_of_features, number_of_features),
    )

    print("generation of correlation matrix ...")
    # first iteration generating correlations to improve the fit of the covariance matrix
    correlation_matrix = correlation_tools.corr_nearest(
        corr=random_matrix, threshold=1e-15, n_fact=100
    )

    # change values on the diagonal to 1 to improve the fit of the covariance matrix
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
    # _visualize_intra_class_correlations(covariant_class)

    # # generate labels
    # label_vector = np.full((number_of_samples, 1), label)
    # correlated_normal_distributed_data = np.hstack((label_vector, covariant_class))
    return covariant_block


def _visualize_intra_class_correlations(data):
    """

    :param data:
    :return:
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
    # pyplot.savefig('C:/Users/sma19/Pictures/correlation_FS/healthy_{}.png'.format(data_name))

    sns.heatmap(corr, cmap="YlGnBu")
    pyplot.show()


def _visualize_distributions(data):
    """

    :param data:
    :return:
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
    number_of_biomarkers,
    number_of_pseudo_class_features,
    number_of_random_features,
):

    # generate label as first entry
    column_names = ["label"]

    # generate names for artificial biomarkers
    for column_name in range(number_of_biomarkers):
        column_names.append("bm_" + str(column_name))

    for column_name in range(number_of_pseudo_class_features):
        column_names.append("pseudo_" + str(column_name))

    for column_name in range(number_of_random_features):
        column_names.append("random_" + str(column_name))

    data_df.columns = column_names
    return data_df


def _generate_classes(
    labels,
    number_of_features,
    number_of_classes,
    number_of_correlated_blocks,
    number_of_samples_per_class,
    scales,
    lower_bounds_for_correlations,
    upper_bounds_for_correlations,
):
    classes = []
    assert len(scales) == number_of_classes
    assert len(labels) == number_of_classes
    # simulation of intraclass correlation
    if number_of_correlated_blocks is not None:
        # generate intraclass correlated classes
        for i, label in enumerate(labels):
            assert number_of_features % number_of_correlated_blocks[i] == 0
            assert number_of_correlated_blocks[i] > 0
            # generate blocks of correlated features
            blocks = []
            for block_number in range(number_of_correlated_blocks[i]):
                # generate block of correlated features
                block = generate_normal_distributed_correlated_block(
                    number_of_features=int(
                        number_of_features / number_of_correlated_blocks[i]
                    ),
                    number_of_samples=number_of_samples_per_class,
                    # mean_list,
                    lower_bound=lower_bounds_for_correlations[i],
                    upper_bound=upper_bounds_for_correlations[i],
                )
                # _visualize_intra_class_correlations(block)
                blocks.append(block)
            generated_class = np.concatenate(blocks, axis=1)
            # _visualize_intra_class_correlations(generated_class)
            assert generated_class.shape[1] == number_of_features
            assert generated_class.shape[0] == number_of_samples_per_class

            # generate labels
            label_vector = np.full((number_of_samples_per_class, 1), label)

            labeled_class = np.concatenate((label_vector, generated_class), axis=1)
            assert labeled_class.shape[1] == number_of_features + 1
            classes.append(labeled_class)
        assert len(classes) == len(number_of_correlated_blocks)

        # generate remaining classes without intraclass correlation
        if len(number_of_correlated_blocks) < number_of_classes:
            for class_label in range(
                number_of_classes - len(number_of_correlated_blocks)
            ):
                normal_distributed_class = generate_normal_distributed_class(
                    label=class_label + len(number_of_correlated_blocks),
                    number_of_samples=number_of_samples_per_class,
                    number_of_biomarkers=number_of_features,
                    scale=scales[class_label + len(number_of_correlated_blocks)],
                )
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
            classes.append(normal_distributed_class)
    assert len(classes) == number_of_classes
    return classes


def generate_artificial_data(
    number_of_normal_distributed_classes=2,
    means_of_normal_distributions=[0, 8],
    scales_of_normal_distributions=[1, 2],
    number_of_lognormal_distributed_classes=3,
    shifts_of_lognormal_distribution_centers=[3, 5, 10],
    number_of_samples_per_class=20,
    number_of_artificial_biomarkers=18,
    # divided by number of blocks and rounded up for simulation of intra class correlations must be vielfaches der blocks
    number_of_intra_class_correlated_blocks_normal_distributed_classes=[3, 3],
    lower_bounds_for_correlations_normal_distributed_classes=np.full(3, 0.7),
    upper_bounds_for_correlations_normal_distributed_classes=np.full(3, 1),
    number_of_intra_class_correlated_blocks_lognormal_distributed_classes=[3, 3, 3],
    lower_bounds_for_correlations_lognormal_distributed_classes=np.full(3, 0.7),
    upper_bounds_for_correlations_lognormal_distributed_classes=np.full(3, 1),
    number_of_pseudo_class_features=4,
    number_of_random_features=50,
    path_to_save_plot=None,
    path_to_save_csv="../data/complete_artif.csv",
    path_to_save_feather=None,
):
    assert (
        len(means_of_normal_distributions)
        == len(scales_of_normal_distributions)
        == number_of_normal_distributed_classes
    ), (
        "The length of the list of means (mean_of_normal_distributions) and "
        "scales (scale_of_normal_distributions) for all normal distributed classes "
        "must match the number of normal distributed classes."
    )
    assert (
        len(shifts_of_lognormal_distribution_centers)
        == number_of_lognormal_distributed_classes
    ), (
        "The length of the list of shifts (shifts_of_lognormal_distribution_centers) for "
        "all lognormal distributed classes "
        "must match the number of lognormal distributed classes."
    )

    # TODO: update to different sample numbers per class
    number_of_classes = (
        number_of_normal_distributed_classes + number_of_lognormal_distributed_classes
    )
    assert number_of_classes >= 2
    number_of_all_samples = number_of_samples_per_class * number_of_classes
    assert number_of_all_samples > 0, "Number of samples must be greater than zero."

    classes = []
    classes_df = pd.DataFrame()

    labels = range(number_of_classes)

    # generate normal distributed classes
    list_of_normal_distributed_classes = _generate_classes(
        labels[:number_of_normal_distributed_classes],
        number_of_artificial_biomarkers,
        number_of_normal_distributed_classes,
        number_of_intra_class_correlated_blocks_normal_distributed_classes,
        number_of_samples_per_class,
        scales_of_normal_distributions,
        lower_bounds_for_correlations_normal_distributed_classes,
        upper_bounds_for_correlations_normal_distributed_classes,
    )
    assert list_of_normal_distributed_classes[0][0, 0] == labels[0]

    for normal_distributed_class, mean_shift in list(
        zip(list_of_normal_distributed_classes, means_of_normal_distributions)
    ):
        assert len(means_of_normal_distributions) == len(
            list_of_normal_distributed_classes
        )
        #  shift class data and exclude the label from shifting
        label = normal_distributed_class[:, 0].reshape(-1, 1)
        shifted_class_data = normal_distributed_class[:, 1:] + mean_shift
        classes_df[str(mean_shift)] = shifted_class_data.flatten()

        labeled_shifted_class = np.hstack((label, shifted_class_data))
        assert labeled_shifted_class[:, 0].all() == label.all()
        classes.append(labeled_shifted_class)
    assert len(classes) == number_of_normal_distributed_classes
    # _visualize_intra_class_correlations(classes_df)

    # generate lognormal distributed classes
    list_of_classes = _generate_classes(
        labels[number_of_normal_distributed_classes:],
        number_of_artificial_biomarkers,
        number_of_lognormal_distributed_classes,
        number_of_intra_class_correlated_blocks_lognormal_distributed_classes,
        number_of_samples_per_class,
        np.ones(number_of_lognormal_distributed_classes),
        lower_bounds_for_correlations_lognormal_distributed_classes,
        upper_bounds_for_correlations_lognormal_distributed_classes,
    )
    for normal_distributed_class, shift_of_lognormal_distribution_center in list(
        zip(list_of_classes, shifts_of_lognormal_distribution_centers)
    ):
        assert len(shifts_of_lognormal_distribution_centers) == len(list_of_classes)
        # If the random variable X is lognormally distributed, then Y=ln(X) has a normal distribution.
        # If Y is normally distributed, then X=exp(Y) is also lognormally distributed.
        # The lognormal distribution has two parameters μ - location and σ - scale.
        # In my example I assumed the simplest approach with μ=0 and σ=1.
        # If you want to have μ=m and σ=s then just put X <- m + s*exp(Y)

        #  shift and transform class data excluding the label
        label = normal_distributed_class[:, 0].reshape(-1, 1)
        shifted_lognormal_distributed_class = (
            np.exp(normal_distributed_class[:, 1:])
            + shift_of_lognormal_distribution_center
        )
        classes_df[
            str(shift_of_lognormal_distribution_center)
        ] = shifted_lognormal_distributed_class.flatten()

        labeled_shifted_lognormal_distributed_class = np.concatenate(
            (label, shifted_lognormal_distributed_class),
            axis=1,
        )
        classes.append(labeled_shifted_lognormal_distributed_class)
    assert len(classes) == number_of_classes
    _visualize_distributions(classes_df.iloc[1:, :])

    complete_classes = np.concatenate(classes, axis=0)

    # check the data shape and the correct generation of the labels
    print(complete_classes.shape)
    assert complete_classes.shape[0] == number_of_samples_per_class * number_of_classes
    assert (
        complete_classes.shape[1] == number_of_artificial_biomarkers + 1
    )  # for the label
    for class_label in labels:
        assert (
            complete_classes[number_of_samples_per_class * class_label, 0]
            == class_label
        )

    # append pseudo class
    pseudo_class = generate_pseudo_class(
        number_of_pseudo_class_features,
        number_of_normal_distributed_classes,
        number_of_lognormal_distributed_classes,
        number_of_samples_per_class,
    )
    complete_classes = np.concatenate((complete_classes, pseudo_class), axis=1)
    assert (
        complete_classes.shape[1]
        == number_of_artificial_biomarkers
        + number_of_pseudo_class_features
        + 1  # for the label
    )

    # append random features
    random_features = np.random.normal(
        loc=0.0, scale=2, size=(number_of_all_samples, number_of_random_features)
    )
    complete_data_set = np.concatenate((complete_classes, random_features), axis=1)

    # check final data shape
    print(complete_data_set.shape)
    assert complete_data_set.shape[0] == number_of_samples_per_class * number_of_classes
    assert (
        complete_data_set.shape[1]
        == number_of_artificial_biomarkers
        + number_of_pseudo_class_features
        + 1  # for the label
        + number_of_random_features
    )

    complete_data_df = pd.DataFrame(complete_data_set)

    # generate feature names
    complete_data_df = _generate_column_names(
        complete_data_df,
        number_of_artificial_biomarkers,
        number_of_pseudo_class_features,
        number_of_random_features,
    )

    # complete_features_df.columns = map(str, complete_features_df.columns)
    # complete_features_df.to_feather(path_feather)
    pd.DataFrame(complete_data_df).to_csv(path_to_save_csv, index=False)
    print(
        "Data generated successfully. You can find the generated file relative to artificial_data.py in: ",
        path_to_save_csv,
    )
    return complete_data_df


def generate_shuffled_artificial_data(
        number_of_normal_distributed_classes=2,
        means_of_normal_distributions=[0, 8],
        scales_of_normal_distributions=[1, 2],
        number_of_lognormal_distributed_classes=3,
        shifts_of_lognormal_distribution_centers=[3, 5, 10],
        number_of_samples_per_class=20,
        number_of_artificial_biomarkers=18,
        # divided by number of blocks and rounded up for simulation of intra class correlations must be vielfaches der blocks
        number_of_intra_class_correlated_blocks_normal_distributed_classes=[3, 3],
        lower_bounds_for_correlations_normal_distributed_classes=np.full(3, 0.7),
        upper_bounds_for_correlations_normal_distributed_classes=np.full(3, 1),
        number_of_intra_class_correlated_blocks_lognormal_distributed_classes=[3, 3, 3],
        lower_bounds_for_correlations_lognormal_distributed_classes=np.full(3, 0.7),
        upper_bounds_for_correlations_lognormal_distributed_classes=np.full(3, 1),
        number_of_pseudo_class_features=4,
        number_of_random_features=50,
        path_to_save_plot=None,
        path_to_save_csv="../data/complete_artif.csv",
        path_to_save_feather=None,
    ):

    complete_data_df = generate_artificial_data(
        number_of_normal_distributed_classes,
        means_of_normal_distributions,
        scales_of_normal_distributions,
        number_of_lognormal_distributed_classes,
        shifts_of_lognormal_distribution_centers,
        number_of_samples_per_class,
        number_of_artificial_biomarkers,
        # divided by number of blocks and rounded up for simulation of intra class correlations must be vielfaches der blocks
        number_of_intra_class_correlated_blocks_normal_distributed_classes,
        lower_bounds_for_correlations_normal_distributed_classes,
        upper_bounds_for_correlations_normal_distributed_classes,
        number_of_intra_class_correlated_blocks_lognormal_distributed_classes,
        lower_bounds_for_correlations_lognormal_distributed_classes,
        upper_bounds_for_correlations_lognormal_distributed_classes,
        number_of_pseudo_class_features,
        number_of_random_features,
        path_to_save_plot,
        path_to_save_csv,
        path_to_save_feather,
    )

    # shuffle artificial features
    column_names = list(complete_data_df.columns[1:])
    random.shuffle(column_names)
    shuffled_column_names = ["label"] + column_names
    shuffled_data_df = complete_data_df[shuffled_column_names]

    assert shuffled_data_df.columns[0] == "label"

    # complete_features_df.columns = map(str, complete_features_df.columns)
    # complete_features_df.to_feather(path_feather)
    pd.DataFrame(shuffled_data_df).to_csv(path_to_save_csv, index=False)
    print(
        "Data generated successfully. You can find the generated file relative to artificial_data.py in: ",
        path_to_save_csv,
    )

    return shuffled_data_df


# generate_artificial_data()

generate_shuffled_artificial_data()

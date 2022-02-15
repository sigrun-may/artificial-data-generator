# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Example for usage of generator for artificial data artificial_data_generator."""
import numpy as np

import artificial_data_generator


parameters = dict(
    number_of_normal_distributed_classes=1,
    means_of_normal_distributions=[0],
    scales_of_normal_distributions=[2],
    scales_of_lognormal_distributions=[1],
    number_of_lognormal_distributed_classes=1,
    shifts_of_lognormal_distribution_centers=[3],
    number_of_samples_per_class=15,
    number_of_features_per_class=30,
    number_of_features_per_correlated_block_normal_dist=0,
    lower_bounds_for_correlations_normal=None,
    upper_bounds_for_correlations_normal=None,
    number_of_features_per_correlated_block_lognormal=[[6, 3, 6]],
    lower_bounds_for_correlations_lognormal=np.full(3, 0.7),
    upper_bounds_for_correlations_lognormal=np.full(3, 1),
    number_of_pseudo_class_features=5,
    number_of_random_features=50,
    path_to_save_plot=None,
    path_to_save_csv="..your_path_to_save.csv",
    path_to_save_feather=None,
)

# generate_artificial_data()
(generated_data_df, meta_data_dict,) = artificial_data_generator.generate_artificial_classification_data(
    number_of_normal_distributed_classes=1,
    means_of_normal_distributions=[0],
    scales_of_normal_distributions=[2],
    scales_of_lognormal_distributions=[1],
    number_of_lognormal_distributed_classes=1,
    shifts_of_lognormal_distribution_centers=[3],
    number_of_samples_per_class=15,
    number_of_features_per_class=30,
    number_of_features_per_correlated_block_normal_dist=None,
    lower_bounds_for_correlations_normal=None,
    upper_bounds_for_correlations_normal=None,
    number_of_features_per_correlated_block_lognormal=[[6, 3, 6]],
    lower_bounds_for_correlations_lognormal=np.full(3, 0.7),
    upper_bounds_for_correlations_lognormal=np.full(3, 1),
    number_of_pseudo_class_features=5,
    number_of_random_features=50,
    path_to_save_plot=None,
    path_to_save_csv="C:/Users/sma19/PycharmProjects/reverse_feature_selection/data/complete_artif.csv",
    path_to_save_feather=None,
    path_to_save_meta_data = "C:/Users/sma19/PycharmProjects/reverse_feature_selection/data/complete_artif.pkl",
    shuffle_features = False,
)
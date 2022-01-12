import biomarker_data_generator
import numpy as np

parameters = dict(
    number_of_normal_distributed_classes=2,
    means_of_normal_distributions=[0, 8],
    scales_of_normal_distributions=[1, 2],
    scales_of_lognormal_distributions=[1, 1],
    number_of_lognormal_distributed_classes=2,
    shifts_of_lognormal_distribution_centers=[3, 5],
    shifts_of_pseudo_classes=[1, 2, 3, 4],
    number_of_samples_per_class=20,
    number_of_artificial_biomarkers=18,
    # divided by number of blocks and rounded up for simulation
    # of intra class correlations must be vielfaches der blocks
    number_of_intra_class_correlated_blocks_normal_distributed=[3, 3],
    number_of_features_per_correlated_block=[[6, 6, 6], [6, 3, 6]],
    lower_bounds_for_correlations_normal=np.full(3, 0.7),
    upper_bounds_for_correlations_normal=np.full(3, 1),
    number_of_intra_class_correlated_blocks_lognormal=[
        3,
        3,
        3,
    ],
    lower_bounds_for_correlations_lognormal=np.full(3, 0.7),
    upper_bounds_for_correlations_lognormal=np.full(3, 1),
    number_of_pseudo_class_features=5,
    number_of_random_features=50,
    path_to_save_plot=None,
    path_to_save_csv="../data/complete_artif.csv",
    path_to_save_feather=None,
)

# generate_artificial_data()
generated_data_df, meta_data_dict = \
    biomarker_data_generator.generate_shuffled_artificial_data(parameters)
biomarker_data_generator.save_meta_data(meta_data_dict,
                                        "../data/complete_artif.pkl")
biomarker_data_generator.save_result(generated_data_df,
                                     "../data/complete_artif.csv")

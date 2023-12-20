import unittest
import numpy as np
import pandas as pd

from artificial_data_generator import artificial_data_generator_tools as adgt


class TestArtificialDataGenerator(unittest.TestCase):
    def test_generate_correlated_cluster_with_valid_inputs(self):
        result = adgt.generate_correlated_cluster(5, 100, 0.1, 0.9)
        self.assertEqual(result.shape, (100, 5))

    def test_generate_correlated_cluster_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            adgt.generate_correlated_cluster(-5, 100, 0.1, 0.9)

    def test_generate_normal_distributed_informative_features_for_one_class_with_valid_inputs(self):
        result = adgt.generate_normal_distributed_informative_features_for_one_class(100, 5, 0.5)
        self.assertEqual(result.shape, (100, 5))

    def test_generate_normal_distributed_informative_features_for_one_class_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            adgt.generate_normal_distributed_informative_features_for_one_class(100, -5, 0.5)

    def test_transform_normal_distributed_class_features_to_lognormal_distribution(self):
        input_data = np.random.normal(0, 1, (100, 5))
        result = adgt.transform_normal_distributed_class_features_to_lognormal_distribution(input_data)
        self.assertEqual(result.shape, input_data.shape)

    def test_shift_class_to_enlarge_effectsize(self):
        input_data = np.random.normal(0, 1, (100, 5))
        result = adgt.shift_class_to_enlarge_effectsize(input_data, 2)
        self.assertEqual(result.shape, input_data.shape)

    def test_build_class(self):
        input_data = [np.random.normal(0, 1, (100, 5)) for _ in range(5)]
        result = adgt.build_class(input_data)
        self.assertEqual(result.shape, (100, 25))

    def test_generate_pseudo_class_with_valid_inputs(self):
        result = adgt.generate_pseudo_class(100, 5)
        self.assertEqual(result.shape, (200, 5))

    def test_generate_pseudo_class_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            adgt.generate_pseudo_class(100, -5)

    def test_generate_random_features(self):
        result = adgt.generate_random_features(100, 5)
        self.assertEqual(result.shape, (100, 5))

    def test_generate_artificial_classification_data_with_valid_inputs(self):
        input_data = [np.random.normal(0, 1, (100, 5)) for _ in range(5)]
        result = adgt.generate_artificial_classification_data(input_data, 100)
        self.assertEqual(result.shape, (500, 6))

    def test_generate_artificial_classification_data_with_invalid_inputs(self):
        with self.assertRaises(ValueError):
            input_data = [np.random.normal(0, 1, (100, 5)) for _ in range(5)]
            adgt.generate_artificial_classification_data(input_data, -100)

    def test_find_perfectly_separated_features_with_valid_inputs(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        class2 = np.array([[7, 8, 9], [10, 11, 12]])
        result = adgt.find_perfectly_separated_features([class1, class2])
        self.assertEqual(result, [0, 1, 2])

    def test_find_perfectly_separated_features_with_no_separated_features(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        class2 = np.array([[1, 2, 3], [4, 5, 6]])
        result = adgt.find_perfectly_separated_features([class1, class2])
        self.assertEqual(result, [])

    def test_find_perfectly_separated_features_with_empty_class(self):
        class1 = np.array([])
        class2 = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            adgt.find_perfectly_separated_features([class1, class2])

    def test_find_perfectly_separated_features_with_one_class(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            adgt.find_perfectly_separated_features([class1])

    def test_find_perfectly_separated_features_with_different_number_of_features(self):
        class1 = np.array([[1, 2, 3], [4, 5, 6]])
        class2 = np.array([[7, 8], [9, 10]])
        with self.assertRaises(ValueError):
            adgt.find_perfectly_separated_features([class1, class2])

    def test_drop_perfectly_separated_features_with_valid_inputs(self):
        data_df = pd.DataFrame({"bm_0": [1, 2, 3], "bm_1": [4, 5, 6], "bm_2": [7, 8, 9]})
        result = adgt.drop_perfectly_separated_features([0, 2], data_df)
        self.assertEqual(result.columns.tolist(), ["bm_1"])

    def test_drop_perfectly_separated_features_with_invalid_separated_features(self):
        data_df = pd.DataFrame({"bm_0": [1, 2, 3], "bm_1": [4, 5, 6], "bm_2": [7, 8, 9]})
        with self.assertRaises(IndexError):
            adgt.drop_perfectly_separated_features([3], data_df)


if __name__ == "__main__":
    unittest.main()

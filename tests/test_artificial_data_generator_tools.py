import unittest
import numpy as np
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


if __name__ == '__main__':
    unittest.main()

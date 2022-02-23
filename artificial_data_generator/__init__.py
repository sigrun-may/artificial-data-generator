# Copyright (c) 2022 Sigrun May,
# Helmholtz-Zentrum für Infektionsforschung GmbH (HZI)
# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data generator main package."""

from artificial_data_generator.artificial_data_generator import (
    generate_artificial_classification_data,
    generate_artificial_data,
    save_meta_data,
    save_result,
)


__version__ = "0.0.1rc3"

__all__ = [
    "generate_artificial_data",
    "generate_artificial_classification_data",
    "save_result",
    "save_meta_data",
    "__version__",
]

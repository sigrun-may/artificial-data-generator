pydocstyle --count biomarker_data_generator
black biomarker_data_generator --check --diff
flake8 biomarker_data_generator
isort biomarker_data_generator --check --diff
mdformat --check README.md
mypy --install-types --non-interactive biomarker_data_generator
pylint biomarker_data_generator
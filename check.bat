pydocstyle --count biomarker-data-generator
black cv_pruner biomarker-data-generator --check --diff
flake8 cv_pruner biomarker-data-generator
isort cv_pruner biomarker-data-generator --check --diff
mdformat --check README.md
mypy --install-types --non-interactive biomarker-data-generator
pylint biomarker-data-generator

pydocstyle --count data_generator
black data_generator --check --diff
flake8 data_generator
isort data_generator --check --diff
mdformat --check README.md
mypy --install-types --non-interactive data_generator
pylint data_generator
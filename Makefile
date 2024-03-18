src := artificial_data_generator
other-src := examples

# check the code
check:
	pydocstyle --count $(src) $(other-src) $(test-src)
	black $(src) $(other-src) $(test-src) --check --diff
	flake8 $(src) $(other-src) $(test-src)
	isort $(src) $(other-src) $(test-src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src) $(other-src) $(test-src)
	pylint $(src) $(other-src)

# format the code
format:
	black $(src) $(other-src) $(test-src)
	isort $(src) $(other-src) $(test-src)
	mdformat *.md

install:
	poetry lock && poetry install --all-extras

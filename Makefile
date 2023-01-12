.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	flake8 .
	mypy --check-untyped-defs .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes tests/

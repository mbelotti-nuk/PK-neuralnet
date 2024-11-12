PYTHON := $(shell command -v python3 2> /dev/null)
PIP = pip3

ifeq ($(PYTHON),)
    PYTHON := python
	PIP := pip
endif


build:
	PYTHON setup.py bdist_wheel sdist
install:
	PYTHON setup.py sdist
	PIP install dist/pkuned-0.1.0.tar.gz
	rm -r pkuned.egg-info/ 
	rm -r .eggs/
	rm -r build
test:
	pytest
	rm -r .pytest_cache/
uninstall:
	PIP uninstall pk_uned

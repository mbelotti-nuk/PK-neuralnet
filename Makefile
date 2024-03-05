build:
	python3 setup.py bdist_wheel sdist
install:
	python3 setup.py sdist
	pip3 install dist/pkdnn-0.1.0.tar.gz
	rm -r dist/ pkdnn.egg-info/ .eggs/
test:
	pytest
	rm -r .pytest_cache/
uninstall:
	pip3 uninstall pkdnn

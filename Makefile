build:
	python.exe setup.py bdist_wheel sdist
install:
	python.exe setup.py sdist
	pip.exe install dist/pkdnn-0.1.0.tar.gz
	rm -r dist/ pkdnn.egg-info/ .eggs/
test:
	pytest.exe
	rm -r .pytest_cache/
uninstall:
	pip.exe uninstall pkdnn

build:
	python.exe setup.py install
	rm -r build/ dist/ pkdnn.egg-info/ .eggs/
install:
	pip.exe install -r requirements.txt

test:
	pytest.exe
	rm -r .pytest_cache/

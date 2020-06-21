clean:
	rm -rfv build/ dist/ *.egg-info
	find . -iname "*.pyc"
	rm -fv .coverage.*

dev-install: clean
	python setup.py develop

.PHONY: run debug test install
run:
	cd src && python3 -m cgolai.cgol

debug:
	cd src && python3 -m cgolai.cgol -cvf test/datafile

test:
	python3 -m unittest discover -s src

install:
	python3 -m pip install .

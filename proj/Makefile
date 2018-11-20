.PHONY: run debug debugf test clean install
run:
	cd src && python3 -m cgolai.cgol

debug:
	cd src && python3 -m cgolai.cgol -vc

debugf:
	cd src && python3 -m cgolai.cgol -cvf src/test/data/debug_file

test:
	python3 -m unittest discover -s src

clean:
	rm -rf cgolai.egg-info

install:
	python3 -m pip install .



.PHONY: test run
run:
	cd src && python3 -m cgolai.cgol

debug:
	cd src && python3 -m cgolai.cgol -vc

debugf:
	cd src && python3 -m cgolai.cgol -cvf data/debug_file

test:
	python3 -m unittest discover -s src

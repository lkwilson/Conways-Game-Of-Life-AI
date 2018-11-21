.PHONY: run debug test install clean
run:
	cd src && python3 -m cgolai.cgol

debug: test
	cd src && python3 -m cgolai.cgol -cvf test/debug.dat

test:
	python3 -m unittest discover -s src

install:
	python3 -m pip install .

clean:
	rm -f src/test/*.dat

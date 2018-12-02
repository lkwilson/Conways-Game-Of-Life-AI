.PHONY: run debug test install clean build
run:
	cd src && python3 -m cgolai.cgol

debug: test
	cd src && python3 -m cgolai.cgol -cvf test/debug.dat

test:
	python3 -m unittest discover -s src

install:
	python3 -m pip install .

install_updatable:
	python3 -m pip install git+git://github.com/larkwt96/cgolai.git

clean:
	rm -f src/test/*.dat

REPORT_OUT="Wilson-Final-Project.tar"
REPORT_FILES="Report.ipynb ReportWritingComponent.docx"
FILES="src proj README.md MANIFEST.in setup.py ${REPORT_FILES}"
build:
	tar cjf ${REPORT_OUT} ${FILES}

.PHONY: all install generate load

all:

install:
	cd .. && make install &>/dev/null

generate: install
	python3 run_report_v2.py

load: install
	echo python3 -m cgolai.cgol -f {the file}

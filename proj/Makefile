.PHONY: all install generate load

all:

install:
	cd .. && make install &>/dev/null

generate: install
	python3 train_nn.py -f data.bat

load: install
	python3 -m cgolai.cgol -f data.bat

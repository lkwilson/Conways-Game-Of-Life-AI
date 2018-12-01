.PHONY: all install generate load

all:

install:
	cd .. && make install &>/dev/null

generate: install
	python3 train_nn.py

load: install
	echo python3 -m cgolai.cgol -f {the file}

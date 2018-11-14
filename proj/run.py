#!/usr/bin/env python3

import os

from cgolai import CgolAi

def main():
    title = "Conway's Game of Life"
    logo = os.path.join('res', 'logo.png')
    size = (600, 400)
    config = {
        'logo':logo,
        'title':title,
        'size':size,
        'boardSize':(e//10 for e in size),
    }
    sys = CgolAi(config)
    sys.run()

if __name__ == "__main__":
    main()

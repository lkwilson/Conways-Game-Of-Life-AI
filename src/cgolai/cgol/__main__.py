import sys
import getopt

from cgolai.cgol import Cgol


def _print_opts():
    print('-v - verbosity')
    print('-c - print controls')
    print('-f filename - file to load')
    print('-w width - width of board')
    print('-h height - height of board')


def _parse_args(config_opts, config_defaults):
    """
    Args:
        config_defaults dict:
            '-{arg}': (Name[, Set Value[, cast]])
            Example: {'-f': ("file", ), '-c': ("ctrl",True)}
        config_opts:
            getopts options argument
            'cf:' for example
    """
    config = {}

    # proc opts
    optlist, args = getopt.getopt(sys.argv[1:], config_opts)
    optlist = dict(optlist)
    for k, v in optlist.items():
        if optlist[k] == '':  # fill default
            optlist[k] = config_defaults[k][1]
        if len(config_defaults[k]) == 3:  # cast
            optlist[k] = config_defaults[k][2](optlist[k])
        config[config_defaults[k][0]] = optlist[k]

    # proc args
    if len(args) > 0:
        raise Exception("wasn't expecting arguments")

    return config


if __name__ == "__main__":
    config_opts = 'vcf:w:h:'
    config_defaults = {
        '-v': ('verbose', True),
        '-c': ('print_controls', True),
        '-f': ('filename', ),
        '-w': ('width', 80, int),
        '-h': ('height', 60, int),
    }
    config = _parse_args(config_opts, config_defaults)
    system = Cgol(**config)
    system.run()

import sys
import getopt

from cgolai.cgol import Cgol


def _print_opts():
    print('-v - verbosity')
    print('-c - print controls')
    print('-f filename - file to load')


def _parse_args(config_opts, config_defaults):
    """
    Args:
        config_defaults dict:
            '-{arg}': (Name[, Set Value])
            {'-f': ("file", ), '-c': ("ctrl",True)}
        config_opts:
            getopts options argument
            'cf:' for example
    """
    config = {}

    # proc opts
    optlist, args = getopt.getopt(sys.argv[1:], config_opts)
    optlist = dict(optlist)
    for k, v in optlist.items():
        if optlist[k] == '':
            optlist[k] = config_defaults[k][1]
        config[config_defaults[k][0]] = optlist[k]

    # proc args
    if len(args) > 0:
        raise Exception("wasn't expecting arguments")

    return config


if __name__ == "__main__":
    config_opts = 'vcf:'
    config_defaults = {
        '-v': ('verbose', True),
        '-c': ('print_controls', True),
        '-f': ('filename', ),
    }
    config = _parse_args(config_opts, config_defaults)
    system = Cgol(**config)
    system.run()

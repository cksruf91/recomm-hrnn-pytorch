import sys


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DefaultDict(dict):
    """KeyError -> return default value"""

    def __init__(self, default, *arg):
        super().__init__(*arg)
        self.default = default

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return self.default


def progressbar(total, i, bar_length=50, prefix='', suffix=''):
    """progressbar
    """
    dot_num = int((i + 1) / total * bar_length)
    dot = '█' * dot_num
    empty = '.' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')

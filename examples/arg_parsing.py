import argparse

__all__ = ['parse_arguments']

def pos_int(s):
    v = int(s)
    if v > 0:
        return v
    else:
        raise argparse.ArgumentTypeError('%r is not a positive integer' % s)


def nonneg_int(s):
    v = int(s)
    if v >= 0:
        return v
    else:
        raise argparse.ArgumentTypeError('%r is not a non-negative integer' % s)


def parse_arguments():
    argParser = argparse.ArgumentParser(
        prog="stick_tetrahedron.py",
        description="Monte-Carlo estimation of probability that 6 segments of a stick randomly broken in 5 places can form a tetrahedron.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argParser.add_argument('-s', '--seed',        default=7777,   type=pos_int,    help="Random seed to initialize algorithms from MT2203 family")
    argParser.add_argument('-b', '--batch_size',  default=65536,  type=pos_int,    help="Batch size for the Monte-Carlo run")
    argParser.add_argument('-n', '--batch_count', default=2048,   type=pos_int,    help="Number of batches executed in parallel")
    argParser.add_argument('-p', '--processes',   default=-1,     type=int,        help="Number of processes used to execute batches")
    argParser.add_argument('-d', '--id_offset',   default=0,      type=nonneg_int, help="Offset for the MT2203/WH algorithms id")
    argParser.add_argument('-j', '--jump_size',   default=0,      type=nonneg_int, help="Jump size for skip-ahead")
      
    args = argParser.parse_args()

    return args

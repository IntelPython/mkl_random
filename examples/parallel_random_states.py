import mkl_random as rnd


def build_MT2203_random_states(seed, id0, n_workers):
    # Create instances of RandomState for each worker process from MT2203 family of generators
    return (rnd.RandomState(seed, brng=('MT2203', id0 + idx)) for idx in range(n_workers))


def build_SFMT19937_random_states(seed, jump_size, n_workers):
    import copy
    # Create instances of RandomState for each worker process from MT2203 family of generators
    rs = rnd.RandomState(seed, brng='SFMT19937')
    yield copy.copy(rs)
    for _ in range(1, n_workers):
        rs.skipahead(jump_size)
        yield copy.copy(rs)


import numpy as np
import mkl_random as rnd

__doc__ = """
Let's solve a classic problem of MC-estimating a probability that 3 segments of a unit stick randomly broken in 2 places can form a triangle. 
Let $u_1$ and $u_2$ be standard uniform random variables, denoting positions where the stick has been broken.

Let $w_1 = \min(u_1, u_2)$ and $w_2 = \max(u_1, u_2)$. Then, length of segments are $x_1 = w_1$, $x_2 = w_2-w_1$, $x_3 = 1-w_2$. 
These lengths must satisfy triangle inequality.

The closed form result is known to be $\frac{1}{4}$.

"""

def triangle_inequality(x1, x2, x3):
    """Efficiently finds `np.less(x1,x2+x3)*np.less(x2,x1+x3)*np.less(x3,x1+x2)`"""
    tmp_sum = x2 + x3
    res = np.less(x1, tmp_sum)   # x1 < x2 + x3
    np.add(x1, x3, out=tmp_sum)
    buf = np.less(x2, tmp_sum)   # x2 < x1 + x3
    np.logical_and(res, buf, out=res)
    np.add(x1, x2, out=tmp_sum)
    np.less(x3, tmp_sum, out=buf) # x3 < x1 + x2
    np.logical_and(res, buf, out=res)
    return res


def mc_dist(rs, n):
    """Monte Carlo estimate of probability on sample of size `n`, using given random state object `rs`"""
    ws = np.sort(rs.rand(2,n), axis=0)
    x2 = np.empty(n, dtype=np.double)
    x3 = np.empty(n, dtype=np.double)

    x1 = ws[0]
    np.subtract(ws[1], ws[0], out=x2)
    np.subtract(1, ws[1], out=x3)
    mc_prob = triangle_inequality(x1, x2, x3).sum() / n

    return mc_prob


def assign_worker_rs(w_rs):
    """Assign process local random state variable `rs` the given value"""
    assert not 'rs' in globals(), "Here comes trouble. Process is not expected to have global variable `rs`"

    global rs
    rs = w_rs
    # wait to ensure that the assignment takes place for each worker
    b.wait()


def worker_compute(w_id):
    return mc_dist(rs, batch_size)


if __name__ == '__main__':
    import multiprocessing as mp
    from itertools import repeat
    from timeit import default_timer as timer

    seed = 77777
    n_workers = 12
    batch_size = 1024 * 256
    batches = 10000

    t0 = timer()
    # Create instances of RandomState for each worker process from MT2203 family of generators
    rss = [ rnd.RandomState(seed, brng=('MT2203', idx)) for idx in range(n_workers) ]
    # use of Barrier ensures that every worker gets one
    b = mp.Barrier(n_workers)

    with mp.Pool(processes=n_workers) as pool:
        # map over every worker once to distribute RandomState instances
        pool.map(assign_worker_rs, rss, chunksize=1)
        # Perform computations on workers
        r = pool.map(worker_compute, range(batches), chunksize=1)

    # retrieve values of estimates into numpy array
    ps = np.fromiter(r, dtype=np.double)
    # compute sample estimator's mean and standard deviation
    p_est = ps.mean()
    pop_std = ps.std()
    t1 = timer()

    dig = 3 - int(np.log10(pop_std))
    frm_str = "{0:0." + str(dig) + "f}"
    print(("Monte-Carlo estimate of probability: " + frm_str).format(p_est))
    print(("Population estimate of the estimator's standard deviation: " + frm_str).format(pop_std))
    print(("Expected standard deviation of the estimator: " + frm_str).format(np.sqrt(p_est * (1-p_est)/batch_size)))
    print("Execution time: {0:0.3f} seconds".format(t1-t0))

import numpy as np
from parallel_mc import parallel_mc_run, sequential_mc_run
from parallel_random_states import build_MT2203_random_states
from sticky_math import mc_six_piece_stick_tetrahedron_prob
from arg_parsing import parse_arguments

def mc_runner(rs):
    return mc_six_piece_stick_tetrahedron_prob(rs, batch_size)

def aggregate_mc_counts(counts, n_batches, batch_size):
    ps = counts / batch_size
    # compute sample estimator's mean and standard deviation
    p_est = ps.mean()
    p_std = ps.std()/np.sqrt(batches)
    
    # compute parameters for Baysean posterior of the probability
    event_count = 0
    nonevent_count = 0
    for ni in counts:
        event_count += int(ni)
        nonevent_count += int(batch_size - ni)

    assert event_count >= 0
    assert nonevent_count >= 0
    return (p_est, p_std, event_count, nonevent_count) 


def print_result(p_est, p_std, mc_size):
    dig = 3 - int(np.log10(p_std)) # only show 3 digits past width of confidence interval
    frm_str = "{0:0." + str(dig) + "f}"

    print(("Monte-Carlo estimate of probability: " + frm_str).format(p_est))
    print(("Population estimate of the estimator's standard deviation: " + frm_str).format(p_std))
    print(("Expected standard deviation of the estimator: " + frm_str).format(np.sqrt(p_est * (1-p_est)/mc_size)))
    print("Total MC size: {}".format(mc_size))
    

if __name__ == '__main__':
    import multiprocessing as mp
    from itertools import repeat
    from timeit import default_timer as timer
    import sys

    args = parse_arguments()
    
    seed = args.seed
    n_workers = args.processes
    if n_workers <= 0:
        n_workers = mp.cpu_count()

    batch_size = args.batch_size
    batches = args.batch_count
    id0 = args.id_offset

    t0 = timer()

    rss = build_MT2203_random_states(seed, id0, n_workers)
    r = parallel_mc_run(rss, n_workers, batches, mc_runner)
    # r = sequential_mc_run(rss, n_workers, batches, mc_runner)

    # retrieve values of estimates into numpy array
    counts = np.fromiter(r, dtype=np.double)
    p_est, p_std, event_count, nonevent_count = aggregate_mc_counts(counts, batches, batch_size)

    t1 = timer()


    print("Input parameters: -s {seed} -b {batchSize} -n {numBatches} -p {processes} -d {idOffset}".format(
        seed=args.seed, batchSize=args.batch_size, numBatches=args.batch_count, processes=n_workers, idOffset=args.id_offset))
    print("")
    print_result(p_est, p_std, batches * batch_size)
    print("")
    print("Bayesian posterior beta distribution parameters: ({0}, {1})".format(event_count, nonevent_count))
    print("")
    print("Execution time: {0:0.3f} seconds".format(t1-t0))

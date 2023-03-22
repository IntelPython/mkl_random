# Parallel Monte-Carlo example

Using `mkl_random` package, we use MT-2203 family of pseudo-random number generation algorithms,
we create workers, assign them RandomState objects with different members of the family of algorithms,
and use multiprocessing Pool to distribute chunks of MC work to them to process.

Each worker gets `rs` and `n` arguments, `rs` representing RandomState object associated with the worker,
and `n` being the size of the problem. `rs` is used to generate samples of size `n`, perform Monte-Carlo
estimate(s) based on the sample and return.

After run is complete, a generator is returns that contains results of each worker. 

This data is post-processed as necessary for the application.

## Stick triangle problem

Code is tested to estimate the probability that 3 segments, obtained by splitting a unit stick 
in two randomly chosen places, can be sides of a triangle. This probability is known in closed form to be $\frac{1}{4}$.

Run python script "stick_triangle.py" to estimate this probability using parallel Monte-Carlo algorithm:

```
> python stick_triangle.py
Parallel Monte-Carlo estimation of stick triangle probability
Parameters: n_workers=12, batch_size=262144, n_batches=10000, seed=77777

Monte-Carlo estimate of probability: 0.250000
Population estimate of the estimator's standard deviation: 0.000834
Expected standard deviation of the estimator: 0.000846
Execution time: 64.043 seconds

```

## Stick tetrahedron problem

Code is used to estimate the probability that 6 segments, obtained by splitting a unit stick in 
5 random chosen places, can be sides of a tetrahedron. 

The probability is not known in closed form. See
[math.stackexchange.com/questions/351913](https://math.stackexchange.com/questions/351913/probability-that-a-stick-randomly-broken-in-five-places-can-form-a-tetrahedron) for more details.

```
> python stick_tetrahedron.py -s 1274 -p 4 -n 8096
Parallel Monte-Carlo estimation of stick tetrahedron probability
Input parameters: -s 1274 -b 65536 -n 8096 -p 4 -d 0

Monte-Carlo estimate of probability: 0.01257113
Population estimate of the estimator's standard deviation: 0.00000488
Expected standard deviation of the estimator: 0.00000484
Total MC size: 530579456

Bayesian posterior beta distribution parameters: (6669984, 523909472)

Execution time: 30.697 seconds

```

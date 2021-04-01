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

## Stick tetrahedron problem

Code is used to estimate the probability that 6 segments, obtained by splitting a unit stick in 
5 random chosen places, can be sides of a tetrahedron. 

The probability is not known in closed form. See
[math.stackexchange.com/questions/351913](https://math.stackexchange.com/questions/351913/probability-that-a-stick-randomly-broken-in-five-places-can-form-a-tetrahedron) for more details.
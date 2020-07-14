## Key Insights

- **Why does differencing actually work to make a non-stationary time series stationary?**
  - insight from Chap 4 HW 3 (week 10): e.g.for a random walk, the absolute values are very different over time, but the **differences between subsequent values are not!**
  - see below, left: random walk, right: 1-times differenced, looks way more stationary

  ![image-20200620191114648](image-20200620191114648.png)





## Summary

### Why are time series interesting to analyze?

### Transformations (logarithmic + Box-Cox)

### Exponential Smoothing

### Holt-Winters Model

### White Noise

### Random Walk

### AR(p)

#### Parameter Estimation (ML, OLS) 

### Bootstrapping

### Stationarity

### Testing for Stationarity (= unit roots)

### Statistical Hypothesis Testing

### Differencing too often (= infinite lag order)

### Partial Autocorrelation

### Checking model residuals for correlation (Ljung-Box tests)

### Checking model residuals for stability (Chow test)

### ARIMA

### Seasonal ARIMA

### VAR(p)

### Cointegration

### VECM

## Definitions

- **lag operator** $L = x_{t-1}$

  - returns the value of one/$k$ time step(s) before t

  - equivalent to backshift operator $\bold B$ 

  - can be used an arbitrary, k times –> $L^kx_t = x_{t-k}$

  - is most importantly used to **calculate autocorrelation of a time series**

  - **lagging a time series reduces n (= number of observations)** 

  - another application: create the lagged time series for model fit using `lm()`

- **difference operator** $\nabla x_t = x_t-x_{t-1}$

  - this gives you the difference between current and previous value 

    –> the resulting time series depicts relative changes and no absolute values!

  - is most importantly **used to make a non-stationary time series stationary**

- **stochastic trend**

  - see Enders, p. 181

- **order of integration $d$** 

  = a time series needs to be differenced $d$ times to be stationary

  - In case we <u>can not</u> reject $H(d-1)$, the order of integration is $d$. 

- **deterministic trend**

- **drift vs. trend**

  - drift = intercept = $\alpha_0$ 
  - (deterministic) trend, indicated by $\beta t$ in the AR(p) model equations

## The Formula Vault

- expected value
- covariance
- correlation
- standard error

- **difference operator** $\nabla x_t = x_t-x_{t-1}\iff \nabla x_t = (1-L)x_t$ 

  - also: $\nabla^2 x_t= \nabla(\nabla x_t)= \nabla x_t-\nabla x_{t-1}$  
  - and: $\nabla^3 x_t = \nabla (\nabla x_t - \nabla x_{t-1})= \nabla^2 x_t - \nabla^2 x_{t-1}$
  - generally speaking: $\nabla^ix_t = (1-L)^ix_t$ 

- **white noise**

  - $x_t = w_t$
  - $\mu=\mathbb{E}[w_t] = 0$
  - $Cov[w_t,w_t] = \sigma^2$ 

- **random walk**

  - $x_t = x_0+\sum_{i=1}^tw_i$  
  - $\mu=\mathbb{E}[w_t] = x_0$ 
  - $\gamma_k(t)=Cov(x_t,x_{t+k})= t\sigma^2$ 
  - $\rho_k(t)= \frac{1}{\sqrt{1+\frac{k}{t}}}$ 

- **random walk with drift**

  - $x_t = \vartheta+x_{t-1}+ w_t$ **this is only the first order!**
  - $\mu=E[X_t]=x_0+\vartheta\cdot t$ 

- **AR(1) process**

  - $x_t = \alpha_0 +\alpha_1x_{t-1}+w_t$
  - $\mu=E[X_t]=^{t\rightarrow \infty}\frac{\alpha_0}{1-\alpha_1}$

- **AR(p) process** (= autoregressive)

- **MA(q) process **(= moving average)

  - $x_t = c_0+w_t+\theta_1w_{t-1}+\dots+\theta_qw_{t-q}$ 

    rewritten as $x_t = c_0+\theta(L)w_t$ 

- **ARMA(p,q) process**

  - $x_t = \alpha_0+\sum_{i=1}^p\alpha_ix_{t-1}+ w_t+\sum_{j=1}^q\theta_jw_{t-j}$ 

    rewritten as $\alpha(L)x_t= \alpha_o+\theta(L)w_t$ 

- ARIMA
- **Seasonal $ARIMA(p,d,q)(P,D,Q)_s$**
- **VAR(p)**
- **VECM (= vector error correction model)**

## Documentation of Key R Functions

- examining a time series object

  - `class()`
  - `start()`
  - `end()`
  - `frequency()` 

- `decompose()`

- `acf()` plots the autocorrelogram of a time series

- `HoltWinters()`

- create white noise: `w <- rnorm(n, sd = 20)`

  - we create it sampling from a normal distribution with mean zero
  - parameters to specify: standard deviation `sd`, number of samples `n`

- log-transform a time series:

- Box-Cox transform a time series: 

- create a random walk:

- create a random walk with drift:

- create AR(1) time series:

- `diff()` 

  - implements the lag/backshift operator

  - option `differences = ` 

    –> How many times should the time series be differenced

  - **used for:** 

    - create the differenced time series as input to **determine the order of integration using the Pantula Principle**

  

- `ar()`

  - input: a time series
  - outputs the $\alpha$ parameters of an AR(p) model that best fits the data

- `polyroot()`

  - outputs the roots of a polynomial, be they real or complex numbers

- Do a Dickey-Fuller test: `ur.df()`

  - `"trend"`
    - most general, unrestricted model: contains
    - test statistic: $\tau_\tau, \phi_3, \phi_2$
  - `"drift"`
    - test statistic: $\tau_\mu, \phi_2$ 
  - `"none"`
    - no intercept and no deterministic trend
    - test statistic: $\tau$

- `pacf()` partial autocorrelation plot

  - 
  - is used to determine the lag order p of an AR(p) process

- […]

- conduct a **Chow breakpoint test**

  - calculate the F statistic with `Fstats`

- calculate the probability of that value to test for the significance of that value with `pf()` 

- `fs$breakpoint` gives you the largest value for the F-statistic in that interval

  = the most unlikely value where the p-value will be smallest!

- `arima.sim()` to generate an ARIMA time series

  - it is not possible to specify a drift/intercept

- `arima(x, order = c(p,d,q))` to estimate the coefficients given that the orders are known

  - `s.e.` are

- […]

*week 12*

- `VAR()` 

  1. to estimate the order p of an VAR(p) process

  1. to fit a model = obtain the coefficients, given p

- `vars::restrict()`

  - removes coefficients from a fitted model that are non-significant

    –> **the p-value is larger than the significance level 0.05 = t-value is below $1.96\approx 2$** 

  - `restrict(fitted_x, method = "ser", thresh = 2.0)`

  - alternatively, this can be done manually with `method = "man"`, but then a `resmat` needs to be specified

    - **the matrix is a binary one**

      –> entries = 0 mean that the coefficient should be set to zero

    - one row for each series

    - where the restriction matrix has one column for each lag term + intercept (xl1, yl1, intercept)

    - values are interpreted by rowk, here row 1 = 1,2,

- Check for Granger causality with `vars::causality()` 

  - from  min

*week 13*

- `ur.ca()`
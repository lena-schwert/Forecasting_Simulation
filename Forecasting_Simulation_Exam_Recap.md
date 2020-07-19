## Key Insights

- **Why does differencing actually work to make a non-stationary time series stationary?**
  
  - insight from Chap 4 HW 3 (week 10): e.g.for a random walk, the absolute values are very different over time, but the **differences between subsequent values are not!**
- see below, left: random walk, right: 1-times differenced, looks way more stationary
  
  ![image-20200620191114648](image-20200620191114648.png)

+ Effect on the variance of random variables when the variables are scaled by a factor, $a$ : since variance is in term of squared factor of the variables, the new variance = $a^2 \times$ old variance

  ```R
  w1 <- rnorm(100000, mean = 0, sd = 2)  
  var(w1) # 4 
  var(2*w1) # 2^2*4 = 16  
  var(0.5*w1) # 0.5^2*4 = 1
  w2 <- rnorm(100000, mean = 0, sd = 1)
  var(0.5*w1+2*w2) # 0.5^2*4 + 2^2+1 = 5
  ```



## Summary

### Why are time series interesting to analyze?

<<<<<<< HEAD
=======
- data = measurements of one or more variables over time, data points have a regular interval, e.g. daily, monthly, quarterly

- if the model is good enough to approximate the ground truth in data, you can make forecasts that will be approximately true

  –> leverage this in business/science/policy scenarios!

- **general assumption: observations from the past can be extrapolated to be used for predictions of the future!**  

>>>>>>> parent of 50d7819... update Lena: recapped chapter 2
### Ergodicity

+ A process is considered ergodic when its statistic properties can be derived from a single and long realization (time series observations) of the hypothetical model 
  + statistic properties: sample moments (mean, variance, skewness & kurtosis)
  + For a single realization, if we manage to stationarize it we'll obtain ergodicity
+ In social science simulation, we always need to assume ergodicity because e.g. real life event can only occur once and we only get a single realization of the process   
+ Egordicity is a sub-class of stationary 
  + All ergodic process must be stationary but not all stationary process are ergodic 
  + a time series with linear trend is not stationary and hence it is not ergodic 
+ Example of an ergodic process: Throwing coins -> we get the same statistic properties of the process when we throw 1000 different coins in one experiment vs. when we throw a single coin repeatedly for 1000 times 
+ Example of a non-ergodic process: Finding the most visited place -> observing the places visited by 1000 different people in a day vs. observing the places a person visited in 1000 days (We'll get different statistic properties!) 

### Decomposition 

+ as a tool to understanding a time series
  + find out the possible cause of variation 
  + figure out the structure of a time seris 
  + prelimary step before selecting/applying a forecasting method 
+ $n_t$: level, $s_t$: seasonal, $r_t$: residuals (should have same variance over time - homoskedasticity) 
+ use `decompose()` function to analyze all the components (trend, seasonal and random)
  + we compare the fit of either the `additive` or `multiplicative` type by analyzing the random component (should look random without any trend and with the same variance over time)
+ 

#### Additive 

+ $n_t + s_t + r_t$; mean of $s_t$ and $r_t$ should be 0  

#### Multiplicative 

<<<<<<< HEAD
+ use it when **seasonal effects tends to increase as the trend increases** 
+ $n_t \cdot s_t \cdot r_t$; mean of $s_t$ and $r_t$ should be 1 
+ if the random variables is modelled by a multiplicative factor & the variable is positive 
  + use log to transform to additive decomposition
=======
$r_t$: residuals (= what is not explained by the other components; should have constant variance over time –> homoskedasticity) 

- **other possible components**
  - trend (= steady in/decrease of the level)
  - cycle (= fluctuations that are more irregular than seasonality, period length might be unknown)

#### Additive Model

+ $n_t + s_t + r_t$; mean of $s_t$ and $r_t$ should be 0 (so the level is overall not influenced)

#### Multiplicative Model

+ $n_t \cdot s_t \cdot r_t$; mean of $s_t$ and $r_t$ should be 1 (so the level is overall not influenced)
>>>>>>> parent of 50d7819... update Lena: recapped chapter 2

#### Alternative Multiplicative 

+ $n_t \cdot s_t + r_t$; mean of $s_t$ should be 1 and mean of $r_t$ should be 0 

<<<<<<< HEAD
### Transformations

#### logarithmic

+ transform multiplicative model to additive model
  
+ $x_t = n_t \cdot s_t \cdot e_t \Rightarrow y_t = ln(x_t) = ln(n_t) + ln(s_t) + ln(e_t)$ 
  
=======
- **other **

### Transformations

#### logarithmic

+ transform multiplicative model to additive model
  + $x_t = n_t \cdot s_t \cdot e_t \Rightarrow y_t = ln(x_t) = ln(n_t) + ln(s_t) + ln(e_t)$ 

>>>>>>> parent of 50d7819... update Lena: recapped chapter 2
+ A well-known example in economics: log return 
  + Return is defined as, $R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}}-1$
  + Gross return is $R_t + 1 = \frac{P_t}{P_{t-1}}$ 
  + log return = log of gross return, $r_t = ln(R_t + 1) = ln(\frac{P_t}{P_{t-1}})=ln(P_t) - ln(P_{t-1})$
    + benefits of using log return: 
      1. if $ln(P_t)$ is a random walk then $\Delta ln(P_t)$ will be stationary (Chap 4 HW 3.R - Apple Stock price) 
      2. log return has nice properties: breakdown the multiplicative model of gross return in $k$ periods into additive model: $[R_t+1]_k = \frac{P_t}{P_{t-1}}\cdot\frac{P_{t-1}}{P_{t-2}}\cdot\frac{P_{t-2}}{P_{t-3}}\dots \frac{P_{t-k+1}}{P_{t-k}} \Rightarrow ln(\frac{P_t}{P_{t-1}}\cdot\frac{P_{t-1}}{P_{t-2}}\cdot\frac{P_{t-2}}{P_{t-3}}\dots \frac{P_{t-k+1}}{P_{t-k}})$
         $\Rightarrow ln((R_t +1)\cdot (R_{t-1}+1)\dots (R_{t-k+1}+1)) = ln(R_t +1) +  ln(R_{t-1}+1)+ \dots + ln(R_{t-k+1}+1)$ 
         $\Rightarrow r_t + r_{t-1} + \dots + r_{t-k+1}$ (summation of log return in $k$ periods = multiplicative of gross return in $k$ periods)  
      3. if returns are independent, then log returns are independent -> uncorrelatedness can be checked with `acf()` and variance of the additive model can be calculated easily:  $var([r_t]_k) = var(r_t) + var(r_{t-1})+ \dots + var(r_{t-k+1})$ but the variance of multiplicative model of gross return is NOT simply $var([R_t+1]_k) = var(R_t+1) \cdot var(R_{t-1}+1) \dots var(R_{t-k+1}+1)$

<<<<<<< HEAD
+ When we apply log transformation, the expectation & variance of the transformed value is not the function of the expectation & variance of the untransformed value! -> due to Jenson's inequality 
  + let $x_t := ln(r_t)$ ~ $N(0, \sigma^2)$ , the transformed value, $r_t = exp(x_t)$ `x = rnorm(1000, sd=2)`, ` r = exp(x)` 
  + `mean(r)` $\neq$ `exp(mean(x))` and `var(r)^0.5` $\neq$ `exp(var(x))^0.5` 
    + the correct formula is `mean(r)` = `exp(var(x)/2)` and `var(r)^0.5` = `(exp(var(x))*(exp(var(x))-1))^0.5`
    + $\mathbb{E}[r_t] = \mathbb{E}[exp(x_t)] \neq exp(\mathbb{E}[x_t]) = 1$ but $\mathbb{E[r_t]} = exp(\sigma^2/2)$
      + Jensen's inequality theorem says for all convex function,$f$ $\Rightarrow \mathbb{E[f(x)]} \ge f(\mathbb{E}[x])$
      + since exponential function is a convex function, $\mathbb{E}[r_t]$ must be larger than 1 
  
  ![log-normal distribution](/Users/jiayan/Documents/GitHub/Forecasting_Simulation/log-normal distribution.png)
  
  + when a normally distributed variables undergo a tranformation with function that is convex, its distribution becomes right skewed (with fat right tail) as the convex function stretches the distribition of $r_t$ as $x_t$ increases -> recall how exponential function looks like (y value increases drastically as x value increase in small scale) 		

=======
>>>>>>> parent of 50d7819... update Lena: recapped chapter 2
#### Box-Cox 

+ $$x_t = B(y_t, \lambda)=\left\{\begin{array}{ll} ln \left(y_{t}\right) & \text { if } \lambda=0 \\ \left(y_{t}^{\lambda}-1\right) / \lambda & \text { otherwise } \end{array}\right.$$
+ to fix the non-normality of the residuals (remove heteroskedasticity/skewness in the residuals)
  + to make the pattern across the data more consistent -> more accurate forecast with data in normality 
  + address limitation of logarithmatic transformation: $y_t$ has to be positive 

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

+ If there exists some linear combinations of some parameters of both nonstationary processes which results in a stationary process. This is called the cointegrating relation.  

+ Cointegrating relation is a **mean reverting process**. (converges to a mean value) The long term forecast of the cointegrated series are linearly related. 

+ Rank$(\Pi)$ tell us the no. of cointegrating relations. 

### VAR 

+ a multivariate model with multiple no. of AR($p$): $X_{t} = A_0 +A_1X_{t-1} +...+ A_pX_{t-p} + R_t$ series up to $k$
$\Rightarrow X_t =A_0 + A_iX_{t-i} + R_t$ where $A_i$ is a $k \times k$ matrix, with coefficients in each row e.g. $\alpha_{11,1}, \alpha_{12,1}, ...,\alpha_{1k,1}$ expresses granger causality of all other series e.g. $x_{2t},...,x_{kt}$ for this corresponding series in the row, e.g. $x_{1t}$ 
$$
\left[\begin{array}{c}
\mathbf{x}_{1t} \\
\mathbf{x}_{2t} \\
\vdots \\
\mathbf{x}_{kt}
\end{array}\right]=\left[\begin{array}{c}
\alpha_1 \\
\alpha_2 \\
\vdots \\
\alpha_k
\end{array}\right]+\left[\begin{array}{ccccc}
\alpha_{11,i} & \alpha_{12,i} & \cdots & \alpha_{1k,i} \\
\alpha_{21,i} & \alpha_{22,i} & \cdots & \alpha_{2k,i} \\
\vdots & \vdots & \ddots & \vdots \\
\alpha_{k1,i} & \alpha_{k2,i} & \cdots & \alpha_{kk,i} \\
\end{array}\right]\left[\begin{array}{c}
\mathbf{x}_{1t-i} \\
\mathbf{x}_{2t-i} \\
\vdots \\
\mathbf{x}_{kt-i}
\end{array}\right]+\left[\begin{array}{c}
\mathbf{r}_{1t} \\
\mathbf{r}_{2t} \\
\vdots \\
\mathbf{r}_{kt} \\
\end{array}\right]
$$
$i = p =$ no. of lags; $k =$ no. of series in the system; first index of $\alpha$ = $k^{th}$ series, second index of $\alpha$ 
+ from this eqn, we can see that no. of parameters of the model: $k+k^2 \times p$
+ no. of lag define the how many $A_i$ matrices we'll have 
+ no. of series define the size of $A_i$ matrices -> always a $k\times k$ matrix 
+ $R_t$ is a zero mean white noise process with a positive definite covariance matrix: $R_t$~ $(0,\Sigma_R)$ where $\Sigma_R = \mathbb{E}[R_t R_t']$
  + $R_t$ can be generated from cholesky decomposition or simply by adding two i.i.d generated white noise and combine with one of the white noise series 

*e.g. for a VAR model with 3 series of lag order=2:*
+ $k=3$ -> $X_t, A_0, R_t$ will be a column vector of 3 ; 2 $A_i$ -> $A_1$, $A_2$ where each is a $3 \times 3$ matrix 
+ total no. of parameters = 3 + 9x2 = 21 parameters 

**To check the stability in a VAR model**: 

+ $\text{det}\left(I_{k}-A_{1} z-\cdots-A_{p} z^{p}\right)=0$ lie outside of the unit circle (> 1 in absolute value)

### VECM

Given a VAR($p$) of I(1): 
$X_t = A_0 + A_1 X_{t-1} + ... + A_p X_{t-p} + R$

There always exists an **error correction** representation of the form:
$\Delta X_{t}=A_{0}+\Pi X_{t-1}+\Gamma_{1} \Delta X_{t-1}+\cdots+\Gamma_{p-1} \Delta X_{t-p+1}+R_{t}$

where
$\Pi=-\left(I_{k}-A_{1}-\cdots-A_{p}\right), \Gamma_{i}=-\left(A_{i+1}+\cdots+A_{p}\right)$
e.g. $k=3, p=2$: `-(diag(3)-A1-A2)` while `A1, A2` is $3 \times 3$ matrices

**Interpretation of VECM:**

+ if $\Pi=0$, all $\lambda(\Pi)=0$, rank=0 -> **no cointegration**; Non-stationary of I(1) vanishes by taking the differences -> we **fit $\Delta X_t$**
+ if $\Pi$ has full rank, $k$, then VAR($p$) is stationary, cannot be I(1) -> **fit VAR model directly**
+ if rank$(\Pi) =m$, $0<m<k$ -> the case of cointegration, we write $\Pi=\alpha\beta'; (k \times k) =(k \times m)[(k \times m)']$ -> **fit VECM($p-1$) model**

### Johansen Test 

+ A procedure to determine the rank of $\Pi$ and whether there is a trend in the cointegrating relations  
  + First, 
+ $H_1^*$ -> $A_0 = \alpha \cdot\beta_0, B=0$: no trend in levels, no trend in cointegrating relations -> **(ecdet="constant")**
`z.vecm<-my.ca.jo(z, type = "trace", spec = "transitory",ecdet="const",K=2)`
+ $H_1$ -> $A_0 \neq 0 , B=0$: linear trend in levels, no trend in cointegrating relations, drift in differences -> **(ecdet="none")**
`z.vecm<-my.ca.jo(z, type = "trace", spec = "transitory",ecdet="none",K=2)`
+ $H^*$ -> $A_0 \neq 0, B= \alpha \cdot \beta_1$: linear trend in levels, linear trend in cointegrating relations, drift in differences -> **(ecdet="trend")**
`z.vecm<-my.ca.jo(z, type = "trace", spec = "transitory",ecdet="trend",K=2)`

#### Checking the cointegrating rank and the presence of trend in the series

+ Check the hypotheses, $H^*_1$ and $H_1$ starting from rank =0 in an alternating order up to the full rank, $k$. 
+ Stop the test when the test statistic is smaller than the significant level (cannot reject the $H_0$) so we accept the $H_0$ which is under the assumption that the rank we are testing is the true rank.  

> When the series has an obvious trend, skip $H_1^*$ and proceed to $H_1$ but we can't know whether the cointegrating relations have a trend, so test $H_1$ against $H^*$

#### Checking whether the cointegrating relations are stationary with no trend

**Testing $H_1$ against $H^*$**: 

+ Hypothesis: no trend in cointegrating relations
+ teststat is chi-square distributed 
+ The likelihood ratio test statistic:
  $$T \sum_{j=1}^{r} \ln \left(\left(1-\lambda_{j}^{1}\right) /\left(1-\lambda_{j}^{*}\right)\right)$$
  + if there is no trend in cointegrating relations, $\lambda^*_j$ will be similar to $\lambda^1_j$ so we will log a value which is close to 1, $ln(1) = 0$ so we'll sum up a value that is close to zero -> test statistic will be small hence cannot reject $H_0$  

## Definitions

- **lag operator** $L = x_{t-1}$

  - returns the value of one/$k$ time step(s) before t

  - equivalent to backshift operator $\mathbf{B}$ 

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

- standard deviation: $\sigma= \sqrt{ \text{variance}}$

- Variance of a sample: $\sum(x-\bar{x})^2/(n-1)$

- expected value

<<<<<<< HEAD
- Covariance($x,y$): $\sum(x-\bar{x})(y-\bar{y})/(n-1)$
=======
- Covariance of two variables ($x,y$): $\sum(x-\bar{x})(y-\bar{y})/(n-1)$
>>>>>>> parent of 50d7819... update Lena: recapped chapter 2

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

- **VAR(p)**: $X_t =A_0 + A_iX_{t-i} + R_t$

  - $k$ is the no. of series in the system
  - $i$ is the no. of lag
  - $X_t$, $X_{t-i}$, $A_0$ and $R_t$ is a $k$-dimension column vector 
  - $A_i$ is a $k \times k$ matrix 

- **VECM (= vector error correction model)**: $\Delta X_{t}=A_{0}+\Pi X_{t-1}+\Gamma_{1} \Delta X_{t-1}+\cdots+\Gamma_{p-1} \Delta X_{t-p+1}+R_{t}$

  + $\Pi=-\left(I_{k}-A_{1}-\cdots-A_{p}\right), \Gamma_{i}=-\left(A_{i+1}+\cdots+A_{p}\right)$
  + $\Pi = \alpha \cdot \beta'$ 
  + cointegrating relations: $\beta'X_{t-i}$

## Documentation of Key R Functions

- examining a time series object

  - `class()`
  - `start()`
  - `end()`
  - `frequency()` 
  - `fix()`: show the structure (data of the time series), Tsp attribute and the class attribute 

- creating a times series object 

  - `y<-structure(c(4,5,6,7), tsp=c(2017.75, 2018.5, 4), class="ts")` 
    - `.Tsp=c(2017.75, 2018.5, 4)` is valid as well 
  - Tsp attribute tells us the start time & end time (in time units) and the frequency: `attr(ts, "tsp")` but not `attr(ts, ".Tsp")`! 
    - start time & end time in time units are calculated as `time + i/f`, where `i` is the period of that time -1 
      - `start(y)` # 2017 4 -> 2017+ 3/4 = 2017.75
      - `end(y)` # 2018 3 ->  2018 + 2/4 = 2018.5
  - convert from other data type to a time series object: `ts()`
    - `ts(x, start = c(2020, 1), freq = 12)`

- Quick check on the trend & seasonal effect of a time series object 

<<<<<<< HEAD
  - `aggregate(ts)`: sum up all observations by each period (e.g. aggegate each month data across multiple years)
  - `cycle(ts)`: give the position in the cycle (e.g. Jan=1, Feb=2, etc.)
  - `window()`: extract all observations of particular period across years e.g. `window(AP, start=c(1949,7), end=c(1957,7) freq=TRUE)`
    - if end argument is not specified -> include up to the last year available in the data

- `decompose()`: default type is addictive; `decompose(ts, type="mult")` 
=======
- `decompose()`
>>>>>>> parent of 50d7819... update Lena: recapped chapter 2

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
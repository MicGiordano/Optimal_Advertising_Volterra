# Optimal_Advertising_Volterra
Computing the optimal advertising effort for simple Volterra-Ornstein-Uhlenbeck dynamics.

This code allows the user to compute the optimal advertising effort U for the optimization problem 

J(u)= E[-\int_0^t U(t)^2 dt + X(T)]

where the dynamics of X(t) are given by the Volterra-Ornstein-Uhlenbeck process

X(t) = X_0 + \int_0^t (t-r)^H (a * U(r) - b * X(r) + MRL) dr + s * \int_0^t (t-r)^H dW(r),  t \in [0,T]

Where W(r) is a standard Brownian motion on [0,T], a, b and s, are positive constants representing the effectiveness of the advertising effort, the speed of memory deterioration in absence of advertising, and the volatility of the model, respectively. 

Here H is the Hurst index, and we know that for 0 < H < 1/2 the model is mean reverting to the mean reversion level MRL.

The script for the computation of the optimal u is a little slow, but working

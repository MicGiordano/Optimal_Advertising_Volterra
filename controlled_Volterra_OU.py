import math as math
from scipy.stats import norm
import numpy as np
import scipy.special as sy
import scipy as sp
import matplotlib.pyplot as plt


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*math.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def Shifted_Legendre(T,n):
    """
    Computes the n-th order shifted Legendre polynomials on the interval [0,T]
    
    Parameters
    ----------
    T : Interval size.
    n : Order of the Legendre Polynomial.

    Returns
    -------
    L : Legendre polynomials matrix.

    """
    L=np.zeros((n+1,n+1))
    Q=np.zeros((n+1,n+1))
    for j in range (0,n+1):
        for i in range (0,n+1):
            if i>=j:
                L[i,j]=((-1)**(i+j))*((sy.factorial(i+j))/(sy.factorial(i-j)*((sy.factorial(j))**2)))
    for i in range(0,n+1):
        Q[i,i]=T**(-i)
    return np.matmul(L,Q)


def polyL_i(L,i,T):
    return lambda x: sum(f(x) for f in [lambda x, j=j: L[i,j]*(x**(j)) for j in range(0,i+1)])
    #Defines the i-th Legendre poly
    
def polyL_i_squared(L,i,T):
    return lambda x: sum(f(x) for f in [lambda x, j=j: L[i,j]*(x**(j)) for j in range(0,i+1)])**2
    #Defines the norm of the i-th Legendre poly
    
def integrand(L,i,H,T):
    return lambda x: sum(f(x) for f in [lambda x, j=j: L[i,j]*(x**(j+H)) for j in range(0,i+1)])
    #Defines the integarnd given by the i-th legendre poly times x^H
        

def Legendre_approximation_coefficients(H,T,n):
    
    """
    Computes the n-th shifted Legendre polynomials approximation coefficients of the deterministic 
    function t^H on the interval [0,T]
    Parameters
    ----------
    H : Hurst index in (0,1/2).
    T : Time orizon .
    n : Order of the Legendre Polynomial .

    Returns
    -------
    C : Coefficient vector

    """
    
    C=np.zeros((1,n+1))
    
    
    L=Shifted_Legendre(T, n)
      
    for i in range(0,n+1):
        C[0,i]=sp.integrate.quad( integrand(L,i,H,T) , 0, T, points=(0))[0]*(sp.integrate.quad(polyL_i_squared(L, i, T),0,T)[0])**(-1)
        
    return C


def approximated_hurst(H,T,n):
    """
    Parameters
    ----------
    H : Hurst index.
    T : Time interval.
    n : Number of iterations .

    Returns
    -------
    C : Coefficients of the Legendre polynomials approximation.

    """
    
    C=np.transpose(Legendre_approximation_coefficients(H,T, n))
    L=Shifted_Legendre(T, n)
    t=np.linspace(0,T,1000)
    A=np.zeros((1000,1000))
    K_e=np.zeros((1,1000))
    K=np.zeros((1,1000))
    K[0,:]=t**H
    for i in range(0,n):
        A[i,:]=polyL_i(L,i,T)(t)*C[i]
        K_e[0,:]=K_e[0,:]+A[i,:]
        
    # plt.plot(t,K_e[0,:]-K[0,:])
    # plt.plot(t,K[0,:])
    # plt.plot(t,K_e[0,:])
    
    return np.linalg.norm(K_e[0,:]-K[0,:])


def Monomial_Coefficients(L,C):
    V=np.zeros(len(L))
    
    for i in range(len(L)):
         for j in range(len(L)):
             V[i]=V[i]+L[j,i]*C[0,j]
            
    return V


def Compute_gammas(n,b,L,C):
    """
    Computes the real numbers "gamma_i^k" needed for computing explicitly the optimal vector U.
    
    Parameters
    ----------
    n : Number of iterations. Must be equal to the number of coefficients computed by the
        Legendre_approximation_coefficients function.
    b : Drift in the Volterra process.
    C : Vector of the coefficients computed via the Legendre_approximation_coefficients.

    Returns
    -------
    Gamma : Elements in the optimal vector U.

    """
    V=Monomial_Coefficients(L, C)
    
    if n == 0:
        Gamma=np.array([1])
        
    else:
        S=0
        G=Compute_gammas(n-1,b,L,C)
        
        for i in range(len(G)):
            S=S+(-b)*G[i]*sy.factorial(i)*V[i]
        Gamma=np.append(S,Compute_gammas(n-1,b,L,C))
        
    return Gamma

def OptimalU(b,H,T,t):
    """
    Computes the optimal advertising effort U

    """
    n=1
    while approximated_hurst(H,T,n)>0.5:
        n=n+1
    C=Legendre_approximation_coefficients(H, T, n)
    L=Shifted_Legendre(T, n)
    
    Kappa=Monomial_Coefficients(L, C)
    
    Gammas=np.zeros((n+1,n+1))
    for i in range(0,n+1):
        Gammas[:,i]=np.append(Compute_gammas(i,b,L,C),np.zeros(n-i))
    
    Times=np.zeros((n+1,1))
    
    for i in range (0,n+1):
        Times[i]=(sy.factorial(i)**(-1))*((T-t)**(i))
        
    S=np.matmul(Gammas,Times)
    
    return np.matmul(Kappa,S)[0]


def Volterra(X_0,a,b,s,H,U=None,N=999,m=1,T=10,Compare=False):
    """
    Generates the Volterra process
    X(t)=X_0+\int_0^t (t-r)^H (a*U(r)-b*X(r))dr+s*\int_0^t (t-r)^H dW(r)
    for t \in [0,T]
    
    Arguments
    ---------
        X_0         is the initial Value
        a>=0        is the advertising effort parameter
        b>=0        is the memory deterioration in absence of advertising parameter
        H           is the Hurst index to be choosen in (0,1/2)
        U           is the advertising strategy. If U is missing the program tries 
                    to use the strategy U(t)=t for t in [2,6] and 0 elsewhere. 
                    If U=optimal, the optimal advertising strategy is computed and
                    the program uses this strategy. Notice that U should be a (N,1)
                    dimensional vector
        N           Is the number of steps that the program has to compute, by
                    default it is set to N=999
        m           is the number of paths to simulate, by default m=1
        T           is the upper bound of the time interval
        Compare     is a boolean parameter that is used to decide whether to compute
                    the path with a=0 in addition to the one with a passed by the user.
                    
    Outputs
    --------
    If Compare=True, Volterra generates as outputs V_a,V_0,t, where t is a vector of size N+1
                     and V_0, V_a are matrices of size (m,N)
                     
    If Compare=False, Volterra generates as outputs V_a,t, where t is a vector of size N+1
                     and, V_a is a matrix of size (m,N)
                     
    """
    # Time step size
    dt = T/N
    # Create an empty array to store the realizations.
    x = np.empty((m,N+1))
    # Initial values of the Brownian motion.
    x[:, 0] = 0
    #Variance of the brownian motion
    delta = 1
    
    brownian(x[:,0], N, dt, delta, out=x[:,1:])
    
    
    t = np.linspace(0, N*dt, N+1)
        
    if U is None:
        if N>600:
            U=np.zeros((1,N))
            for i in range(200,600):
                U[0,i]=t[i]-t[200]
        else:
            U=np.zeros((1,N))
            
    elif U=='optimal':  
        U=np.zeros((1,N))
        for i in range(0,N):
            U[0,i]=OptimalU(b,H,T,t[i])
    #If 'optimal' is passed as an argument, the program computes the optimal vector U 
        
    
    #Initialize some empty vectors used in the for cycles
    S1=np.zeros((m,N)) 
    S2=np.zeros((m,N))
    S3=np.zeros((m,N))
    P1=np.zeros((m,N))
    P2=np.zeros((m,N))
    Noise=np.zeros((m,N))
    V_a=np.zeros((m,N))
    V_0=np.zeros((m,N))
 
    #Compute the noise
    for k in range(m):
      for j in range(0,N):
        for i in range (0,j-1):
            S1[k,i]=(s*(t[j]-t[i])**H)*(x[k,i+1]-x[k,i])      
            Noise[k,j]=Noise[k,j]+S1[k,i]
    
    
    if Compare is True:     #Compute the Volterra process both with 'a' and without it (V_a vs V_0)
        for k in range(m):
          V_a[k,0]=X_0
          V_0[k,0]=X_0
          for j in range(1,N):
            for i in range(0,j):
              S2[k,i]=(((t[j]-t[i])**H))*((-b*V_a[k,i]+a*U[0,i])*dt)
              P1[k,j]=P1[k,j]+S2[k,i]
              S3[k,i]=(((t[j]-t[i])**H))*((-b*V_0[k,i])*dt)
              P2[k,j]=P2[k,j]+S3[k,i]
            V_a[k]=V_a[k,0]+P1[k]+Noise[k]
            V_0[k]=V_0[k,0]+P2[k]+Noise[k]     
        
        
        plt.plot(t[0:N],V_a[0,:])
        plt.plot(t[0:N],V_0[0,:])
        plt.plot(t[0:N],U[0,:])
            
        return V_a,V_0,t,U
    
    else:                   #Compute the Volterra process only with 'a' selected by the user
        for k in range(m):
          V_a[k,0]=X_0
          for j in range(1,N):
            for i in range(0,j):
              S2[k,i]=(((t[j]-t[i])**H))*((-b*V_a[k,i]+a*U[0,i])*dt)
              P1[k,j]=P1[k,j]+S2[k,i]
            V_a[k]=V_a[k,0]+P1[k]+Noise[k]
            
        return V_a,t
    


        
    
    



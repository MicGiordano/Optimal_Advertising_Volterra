import math as math
from scipy.stats import norm
import numpy as np
import scipy.special as sy
import scipy as sp
import matplotlib.pyplot as plt
from r8_choose import r8_choose
from r8_mop import r8_mop


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


def approximated_hurst_bernstein(T,H,n):
    """
    Parameters
    ----------
    H : Hurst index.
    T : Time interval.
    n : Number of iterations .

    Returns
    -------
    C : Coefficients of the Bernstein polynomials approximation.

    """
    t=np.linspace(0,T,1000)
    K=np.zeros((1,1000))
    K[0,:]=t**H
    A=np.zeros((1,n+1))
    K_e=np.zeros((1,1000))
    for j in range (len(t)):
        for i in range(0,n+1):
            A[0,i]=(((i*T)/n)**H)*r8_choose(n,i)*(t[j]**i)*(T-t[j])**(n-i)
            K_e[0,j]=K_e[0,j]+(1/(T**n))*A[0,i] 
            
    
    # plt.plot(t,K[0,:])
    # plt.plot(t,K_e[0,:])
    # plt.plot(t,K_e[0,:]-K[0,:])
    
    return np.linalg.norm(K[0,:]-K_e[0,:],ord=np.inf)




def Monomial_Coefficients_Bernstein(T,H,n):
    
    """
    Given the matrix L and the coefficients C, computes the Vector of the monomial coefficients
    for the Legendre polynomials

    Parameters
    ----------
    H : Hurst index.
    T : Time interval.
    n : Number of iterations .

    Returns
    -------
    V : Monomial coefficients for the approximant of t^H

    """
    V=np.zeros((1,n+1))
    A=np.zeros((1,n+1))
    for k in range(0,n+1):
        for i in range (0,k+1):
            A[0,i]=((-1)**(k-i))*(((i*T)/n)**H)*r8_choose(n,i)*r8_choose(n-i,k-i)
            V[0,k]=V[0,k]+((1/T)**k)*A[0,i]    
    return V



def Compute_Gammas_Bernstein(b,T,H,m,V):
    """
    Computes the real numbers "gamma_i^k" needed for computing explicitly the optimal vector U.
    
    Parameters
    ----------
    m : Order of truncation for infinite series
    b : Drift in the Volterra process.
    T : Time horizon
    H : Hurst index

    Returns
    -------
    Gamma : Elements in the optimal vector U.
    """
    
    if m == 0:
        Gamma=np.array([1])
        
    else:
        S=0
        G=Compute_Gammas_Bernstein(b,T,H,m-1,V)
        
        for i in range(min(len(G)-1,len(V[0])-1)+1):
            S=S+(-b)*G[i]*sy.factorial(i)*V[0,i]
            
        Gamma=np.append(S,G)
        
        
    return Gamma

def OptimalU(a,b,s,H,T,t,m,Error=0.5):
    """
    Computes the optimal advertising effort U at point t\in[0,T]. In order to do so we start by 
    computing the optimal advertising effort for an approximated problem with kernel K_e given
    by the Legendre polynomial approximation of K. 
    
    Parameters:
    -------
    a,b,s,H,T : Parameters of the Volterra-OU process
    t         : Point at which computin the optimal advertising effort
    m         : Order of approximation for the truncation of an infinite sum
    Error     : Approximation error for the kernel K_e
    
    Output:
    ---------
    
    U(t)     : Optimal advertising effort at point U(t), t\in [0,T]

    """
    

    n=1
    
    while approximated_hurst_bernstein(T,H,n)>Error:         # Finding an optimal number of Legendre poly that allows to get below the approximation error choosen by the user
        n=n+1
    
    Kappa=Monomial_Coefficients_Bernstein(T,H,n)
    
    Gammas=np.zeros((m+1,m+1))
    
    
    for i in range(0,m+1):
        Gammas[:,i]=np.append(Compute_Gammas_Bernstein(b,T,H,i,Kappa),np.zeros(m-i))
    
    Times=np.zeros((m+1,1))
    
    for i in range (0,m+1):
        Times[i]=(sy.factorial(i)**(-1))*((T-t)**(i))
        
    S=np.matmul(Gammas,Times)
    
    if m>n:
        return (np.matmul(Kappa,S[0:n+1])[0]**2)*(a/2)
    else:
        return (np.matmul(Kappa[0,0:m+1],S)[0])*((1/2)*(a))
    
    


def Volterra(X_0,a,b,s,H,U=None,N=999,m=20,T=10,Compare=False,Error=0.5):
    """
    Generates the Volterra process
    X(t)=X_0+\int_0^t (t-r)^H (a*U(r)-b*X(r))dr+s*\int_0^t (t-r)^H dW(r)
    for t \in [0,T].
    
    
    Parameters
    ---------
        X_0   :      Is the initial Value
        a>=0  :      Is the advertising effort parameter
        b>=0  :      Is the memory deterioration in absence of advertising parameter
        H     :      Is the Hurst index to be choosen in (0,1/2)
        U     :      Is the advertising strategy. If U is missing the program tries 
                     to use the strategy U(t)=t for t in [2,6] and 0 elsewhere. 
                     If U=optimal, the optimal advertising strategy is computed and
                     the program uses this strategy. Notice that U should be a (N,1)
                     dimensional vector
        N     :      Is the number of steps that the program has to compute, by
                     default it is set to N=999
        m     :      Order of truncation for the approximated exponent 
        T     :      Is the upper bound of the time interval
        Compare:     Is a boolean parameter that is used to decide whether to compute
                     the path with a=0 in addition to the one with a passed by the user.
        Error:       Is the approximation error allowed when computing the approximant polynomial kernel
                     This is used in computing the optimal advertising effort U
        Method:      Uses either Legendre or Bernstein Polynomial approximation
        
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
    x = np.empty((1,N+1))
    # Initial values of the Brownian motion.
    x[:, 0] = 0
    #Variance of the brownian motion
    delta = 1
    
    brownian(x[:,0], N, dt, delta, out=x[:,1:])
    
    
    t = np.linspace(0, N*dt, N+1)
        
    if U is None:           #If no U is used the default vector is a zeroes vector
        U=np.zeros((1,N))
        
    elif U == 'test':      #If 'test' is chosen, and N is large enough, a test vector is choosen for U
        if N>600:
            U=np.zeros((1,N))
            for i in range(200,600):
                U[0,i]=t[i]-t[200]
        else:
            U=np.zeros((1,N))
            
    elif U=='optimal':     #If 'optimal' is passed as an argument, the program computes the optimal vector U 
        U=np.zeros((1,N))
        for i in range(0,N):
            U[0,i]=OptimalU(a,b,s,H,T,t[i],m,Error)
   
    
    #Initialize some empty vectors used in the for cycles
    S1=np.zeros((1,N)) 
    S2=np.zeros((1,N))
    S3=np.zeros((1,N))
    P1=np.zeros((1,N))
    P2=np.zeros((1,N))
    Noise=np.zeros((1,N))
    V_a=np.zeros((1,N))
    V_0=np.zeros((1,N))
 
    #Compute the noise
    for j in range(0,N):
      for i in range (0,j-1):
          S1[0,i]=(s*(t[j]-t[i])**H)*(x[0,i+1]-x[0,i])      
          Noise[0,j]=Noise[0,j]+S1[0,i]
    
    
    if Compare is True:     #Compute the Volterra process both with 'a' and without it (V_a vs V_0)
        V_a[0,0]=X_0
        V_0[0,0]=X_0
        for j in range(1,N):
          for i in range(0,j):
            S2[0,i]=(((t[j]-t[i])**H))*((-b*V_a[0,i]+a*U[0,i])*dt)
            P1[0,j]=P1[0,j]+S2[0,i]
            S3[0,i]=(((t[j]-t[i])**H))*((-b*V_0[0,i])*dt)
            P2[0,j]=P2[0,j]+S3[0,i]
          V_a[0]=V_a[0,0]+P1[0]+Noise[0]
          V_0[0]=V_0[0,0]+P2[0]+Noise[0]     
        
        
        # plt.plot(t[0:N],V_a[0,:],label='X(t), alpha=1')
        # plt.plot(t[0:N],V_0[0,:],label='X(t), alpha=0')
        # plt.plot(t[0:N],U[0,:],label='u(t)')
        # plt.legend(loc="upper right")
        # plt.show()
        # plt.savefig('Volterra.png', dpi=300)
                
        return V_a,V_0,t,U
    
    else:                   #Compute the Volterra process only with 'a' selected by the user
        V_a[0,0]=X_0
        for j in range(1,N):
            for i in range(0,j):
                S2[0,i]=(((t[j]-t[i])**H))*((-b*V_a[0,i]+a*U[0,i])*dt)
                P1[0,j]=P1[0,j]+S2[0,i]
            V_a[0]=V_a[0,0]+P1[0]+Noise[0]
            
        return V_a,t
    

        
    
    



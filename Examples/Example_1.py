from controlled_Volterra_OU import Volterra

a=1
X_0=0
b=1
s=1
MRL=0
H=0.3


V_a,V_0,t=Volterra(X_0,a,b,s,MRL,H,Compare=True)

"""

In this example we consider the standard Volterra-Ornstein–Uhlenbeck process, with parameters as above. 
The main point is to show the difference between the advertised model V_a and the non-advertised model V_0.

Here we consider the default advertising vector u, defined as u(t) = t for t in [2,6] and 0 elsewhere.

"""

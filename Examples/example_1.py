from controlled_Volterra_OU import Volterra

a=1
X_0=0
b=1
s=1
MRL=0
H=0.3


V_a,V_0,t=Volterra(X_0,a,b,s,MRL,H,Compare=True)

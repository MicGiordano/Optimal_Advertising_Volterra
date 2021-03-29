from controlled_Volterra_OU import Volterra

a=10
b=10
X_0=0
s=100
H=0.3


V_a,V_0,t,U=Volterra(X_0,a,b,s,H,T=2,Compare=True,U='optimal',Error=0.5)




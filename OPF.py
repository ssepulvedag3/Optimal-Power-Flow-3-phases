import numpy as np
import networkx as nx
import cvxpy as cv
import pandas as pd 
import time

start = time.time()

Sbase = 1000
Vbase = 11/np.sqrt(3)
Zbase = ((1000*Vbase)**2)/(1000*Sbase)

Lines=pd.read_excel('25nodes.xlsx',sheet_name="Lines")
Nodes=pd.read_excel('25nodes.xlsx',sheet_name="Nodes")
Nodes.iloc[:,2:8] = Nodes.iloc[:,2:8]/Sbase


def Matriz8Z(C):
  if C == 1:
    Zm = np.array([[0.093654+0.040293j,    0.031218+0.013431j,  0.031218+0.013431j],
          [0.031218+0.013431j,  0.093654+0.040293j, 0.031218+0.013431j],
          [0.031218+0.013431j,  0.031218+0.013431j, 0.093654+0.040293j]])
      
  elif C == 2:        
    Zm = np.array([[0.15609+0.067155j,  0.05203+0.022385j,  0.05203+0.022385j],
          [0.05203+0.022385j,   0.15609+0.067155j,  0.05203+0.022385j],
          [0.05203+0.022385j,   0.05203+0.022385j,  0.15609+0.067155j]])
    
  elif C == 3:
    Zm =np.array([[0.046827+0.0201465j, 0.015609+0.0067155j,     0.015609+0.0067155j],
          [0.015609+0.0067155j, 0.046827+0.0201465j,     0.015609+0.0067155j],
          [0.015609+0.0067155j, 0.015609+0.0067155j,     0.046827+0.0201465j]]) 
    
  elif C == 4:       
    Zm = np.array([[0.031218+0.013431j, 0.010406+0.004477j, 0.010406+0.004477j],
        [0.010406+0.004477j,    0.031218+0.013431j, 0.010406+0.004477j],
        [0.010406+0.004477j,    0.010406+0.004477j, 0.031218+0.013431j]])

  elif C == 5:      
    Zm = np.array([[0.062436+0.026862j, 0.020812+0.008954j, 0.020812+0.008954j],
          [0.020812+0.008954j,  0.062436+0.026862j, 0.020812+0.008954j],
          [0.020812+0.008954j,  0.020812+0.008954j, 0.062436+0.026862j]])
  elif C == 6:       
    Zm = np.array([[0.078045+0.0335775j,    0.026015+0.0111925j,     0.026015+0.0111925j],
        [0.026015+0.0111925j,   0.078045+0.0335775j,     0.026015+0.0111925j],
        [0.026015+0.0111925j,   0.026015+0.0111925j,     0.078045+0.0335775j]])
     

  return Zm

#Formation of the Incidence Matrix. 

NN=len(Nodes)
NL = len(Lines)
Zr = np.zeros((3*NL,3*NL),dtype="complex")

#Matriz incidencia Nodo-Rama
A = np.zeros((3*NN,3*NL))

l = 0
n = 0

A = np.zeros((3*NN,3*NL))
l = 0
n = 0
for i in range(NL):
    Ni = Lines['from '][i]
    Nj = Lines['to'][i]
    for j in range(3):
        for k in range(NN):
            for p in range(3):             
                if k == Ni-1 and p == j:
                  A[n,l] = 1 
                elif k == Nj-1 and p == j:
                  A[n,l] = -1 
                n = n + 1 
        l = l + 1
        n = 0       
    Zr[3*i:3*(i+1),3*i:3*(i+1)]=(Matriz8Z(Lines['Zm'][i]))/Zbase

Yr=np.linalg.inv(Zr)

Ybus=A@Yr@A.T


M3=np.zeros((3*NN,3*NN))
Z3=np.zeros((3*NN,3*NN))
L3=np.zeros((3*NN,3*NN))

Mp=np.array([[1,-1,0],[0,1,-1],[-1,0,1]])
Zp=np.array([[0,-1,0],[0,0,-1],[-1,0,0]])
Lp=np.array([[0,1,0],[0,0,1],[1,0,0]])

for k in range(NN):
  M3[3*k:3*k+3,3*k:3*k+3] = Mp
  Z3[3*k:3*k+3,3*k:3*k+3] = Zp
  L3[3*k:3*k+3,3*k:3*k+3] = Lp


##Optimal Power Flow 

V_o=np.ones((3*NN,1),dtype="complex")
d_y=np.zeros((3*NN,1),dtype="complex")
d_d=np.zeros((3*NN,1),dtype="complex")
smax=np.zeros((3*NN,1))
smax[0]=50
smax[1]=50
smax[2]=50

smax[18]=0.04
smax[19]=0.05
smax[20]=0.03
smax[33]=0.12
smax[51]=0.055
smax[52]=0.055
smax[53]=0.055
smax[72]=0.04
smax[73]=0.03


for k in range(NN):
  #tensiones
  V_o[3*k]=1
  V_o[3*k+1]=1*np.exp(1j*-120*np.pi/180)
  V_o[3*k+2]=1*np.exp(1j*120*np.pi/180)
  #potencias
  if Nodes['TipoL'][k]==0:
    d_y[3*k]=Nodes['Pa'][k]+Nodes['Qa'][k]*1j
    d_y[3*k+1]=Nodes['Pb'][k]+Nodes['Qb'][k]*1j
    d_y[3*k+2]=Nodes['Pc'][k]+Nodes['Qc'][k]*1j
  if Nodes['TipoL'][k]==1:
    d_d[3*k]=Nodes['Pa'][k]+Nodes['Qa'][k]*1j
    d_d[3*k+1]=Nodes['Pb'][k]+Nodes['Qb'][k]*1j
    d_d[3*k+2]=Nodes['Pc'][k]+Nodes['Qc'][k]*1j

for k in range(10):
  v=cv.Variable((3*NN,1),complex=True)
  s=cv.Variable((3*NN,1),complex=True)
  res=[v[0]==1]
  res+=[v[1]==1*np.exp(1j*-120*np.pi/180)]
  res+=[v[2]==1*np.exp(1j*120*np.pi/180)]
  delta_conection=np.diagflat(V_o)@M3.T@np.linalg.inv(np.diagflat(M3@V_o))@d_d+(M3.T@(np.linalg.inv(np.diagflat(M3@V_o))**2))@(np.diagflat(V_o)@np.diagflat(d_d)@L3+np.diagflat(Z3@V_o)@np.diagflat(d_d))@v
  res+=[s-delta_conection==np.diagflat(V_o)@np.conjugate(Ybus)@cv.conj(v)-np.diagflat(V_o)@np.conjugate(Ybus@V_o)+np.diagflat(np.conjugate(Ybus@V_o))@v+d_y]
  res+=[cv.abs(s)<=smax]
  res+=[cv.real(s)>=0]
  obj=cv.Minimize(cv.real(cv.sum(s)-np.sum(d_y)))
  OPF=cv.Problem(obj,res)
  OPF.solve(solver=cv.ECOS)
  V_o=v.value
  print((V_o.T@np.conjugate(Ybus@V_o))*1000)
print(obj.value*1000)

end = time.time()
print(end-start)
#__author__="Ali Rad"

from mindfoundry.optaas.client.client import OPTaaSClient, Goal
from mindfoundry.optaas.client.result import Result
from mindfoundry.optaas.client.constraint import Constraint
from mindfoundry.optaas.client.parameter import IntParameter, FloatParameter, CategoricalParameter,BoolParameter, ChoiceParameter, GroupParameter
client = OPTaaSClient('https://optaas.mindfoundry.ai/', your API Key)# Connect to your OPTaaS server with your API Key

import numpy as np
from Core_Definition import *
from Hybrid_Optimizer_OPTaaS import * #if loaded Hybrid_Optimizer_OPTaaS, the optimization will be done with OPTaaS service. If loaded Hybrid_Optimizer, the optimization will be done with Gradient Descent
from Auto_Algorithm import *
from Visualization import *
import os
import itertools
import time
n,m=[2,3]

bas=Hybrid_Quantum_Circuit(n*m,"BAS_All_to_All") # size, name | by runing this two directories will be created

def Ideal_state(n,m):
  state_Ideal=state_Ideal=[0]*2**(n*m)


  for i in range(0,m+1):
    res=list(itertools.combinations(range(m), i))
    l=len(res)
    for j in range(l):

      sum=0
      for k in range(i):
        for q in range(n):
          sum=sum+2**(res[j][k]+q*m)
      state_Ideal[sum]=1
  for i in range(0,n+1):
    res_row=list(itertools.combinations(range(n), i))
    l_row=len(res_row)
    for j in range(l_row):

      sum=0
      for k in range(i):
        for q in range(m):
          sum=sum+2**(res_row[j][k]*m+q)
      state_Ideal[sum]=1
  return state_Ideal/np.sum(state_Ideal)

def Score_BAS(state):
    #n,m=[2,2]
    ideal=Quantum_State(n*m)
    ideal.population=Ideal_state(n,m)
    #return CNLL(state,ideal)
    return KL_Divergence(state,ideal)

XX_combin=list(itertools.combinations(range(n*m), 2))
for i in range(len(XX_combin)):
  Define_Parameter("XX"+str(XX_combin[i][0])+str(XX_combin[i][1]),0)

for i in range(n*m):
    Define_Parameter("X"+str(i),0)
    Define_Parameter("Z"+str(i),0)

for i in range(n*m):
  bas.Add_Gate(Quantum_Gate("SKAX",0,angle="X"+str(i)))
  bas.Add_Gate(Quantum_Gate("AZ",0,angle="Z"+str(i)))

for i in range(len(XX_combin)):
  bas.Add_Gate(Quantum_Gate("XA",XX_combin[i][0],XX_combin[i][1],angle="XX"+str(XX_combin[i][0])+str(XX_combin[i][1])))


configurations,current_file=bas.Initialize_Optimization(1)
ions=Auto_Algorithm(n*m)
for i in range(Switching_point=80):
    ions.Load_Algorithm_List(current_file)
    ions.Run("BAS_All_to_All_Readout/",1000)
    configurations,current_file=bas.Update_Result(configurations,Score_BAS)



params_history=dict()
functions_history=dict()
iterations=0
log='BAS_All_to_All_log_run1_two_layer_star.txt'
read_file=open(log,"r")
while True:
        temp=read_file.readline().split(":")
        if len(temp[0])==0:
            break
        if len(temp)==0:
            break
        current_index=int(temp[0])
        if temp[1]=="parameter":
            if temp[2] in params_history:
                if len(params_history[temp[2]])>current_index:
                    params_history[temp[2]][current_index]=float(temp[3])
                else:
                    params_history[temp[2]].append(float(temp[3]))
            else:
                params_history[temp[2]]=[float(temp[3])]

        elif temp[1]=="function":
            if temp[2] in functions_history:
                if len(functions_history[temp[2]])>current_index:
                    functions_history[temp[2]][current_index]=float(temp[3])
                else:
                    functions_history[temp[2]].append(float(temp[3]))
            else:
                functions_history[temp[2]]=[float(temp[3])]

for key in functions_history:
        length=len(functions_history[key])

def f(x):

  new_param=x
  i=0
  for key in param_table:
        Define_Parameter(key,new_param[i])
        param_table[key]=new_param[i]
        i=i+1
  circuit=bas.GatesLab_Sequence()
  temp=Quantum_Circuit(n*m,"test")
  temp.Interpret_GatesLab_Sequence(circuit)
  result=temp.Simulate()
  return Score_BAS(result)
def hist(x):
  new_param=x
  i=0
  for key in param_table:
        Define_Parameter(key,new_param[i])
        param_table[key]=new_param[i]
        i=i+1
  circuit=bas.GatesLab_Sequence()
  temp=Quantum_Circuit(n*m,"test")
  temp.Interpret_GatesLab_Sequence(circuit)
  result=temp.Simulate()

  tot=len(result.state)
  width=0.8/tot
  shift=0
  value=result.population
  xindex=np.arange(len(value))+shift
  pllt=plt.bar(xindex-0.2,value)
  pllt2=plt.bar(xindex+0.2, Ideal_state(n,m))
  plt.show()

cost_history=[]
params_history=[]
time_history=[]
import time
start=time.time()
def cost_execution(params):

    global iterations

    cost_value =f(params)
    cost_history.append(cost_value)
    params_history.append(params)
    now=time.time()-start
    time_history.append(now)

    if iterations % 5 == 0:
        print("Cost at Step {}: {}".format(iterations, cost_value))

        print(params)
        hist(params)
    iterations += 1
    return cost_value

iterations = 0
ini_param_BO=params_history[-1]
minimize(cost_execution, x0=ini_param_BO, method="Nelder-Mead", options={'disp': True})

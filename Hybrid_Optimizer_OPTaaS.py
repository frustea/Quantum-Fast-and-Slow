#__author__="Daiwei Zhu"


import numpy as np
import os
from Core_Definition import *
from mindfoundry.optaas.client.client import OPTaaSClient, Goal
from mindfoundry.optaas.client.result import Result
from mindfoundry.optaas.client.constraint import Constraint
from mindfoundry.optaas.client.parameter import IntParameter, FloatParameter, CategoricalParameter,BoolParameter, ChoiceParameter, GroupParameter
import datetime
import time
start=time.time()
# Connect to your OPTaaS server with your API Key
client = OPTaaSClient('https://edu.optaas.mindfoundry.ai', Your Key)


def KL_Divergence(state1,state0):
    eps=[0.0001]*(2**state1.size)
    q=np.maximum(eps,state1.population)
    p=np.maximum(eps,state0.population)
    temp=np.sum(np.multiply(q,np.log(np.divide(q,p))))
    return temp

def CNLL(state,ideal):
    eps=[0.000001]*(2**state.size)
    p=np.maximum(eps,state.population)
    temp=-np.multiply(ideal.population,np.log(p))
    return np.sum(temp)

def JS_Divergence(state1,state0):
    temp_state=Quantum_State(state1.size)
    temp_state.population_only=True
    for i in range(len(state1.population)):
        temp_state.population[i]=(state1.population[i]+state0.population[i])/2

    return 0.5*KL_Divergence(state1,temp_state)+0.5*KL_Divergence(state0,temp_state)

def To_OPTaaS_Params():#generate OPTaaS parameters from param_table
    temp=[]
    for key in param_table:
        if key.find("xx")>-1:
            temp.append(FloatParameter(key, minimum=-0.5, maximum=0.5,cyclical=True))
        elif True:
            temp.append(FloatParameter(key, minimum=-1,maximum=1,cyclical=True))
    return temp

def From_OPTaaS_Params(configuration):#translate OPTaaS configurations to param_table
    for key in configuration.values:
        Define_Parameter(key,configuration.values[key])

class Hybrid_Quantum_Circuit(Quantum_Circuit): #A quantum circuit class that is designed for hybrid optimization
    def __init__(self,size,name="current"):
        self.size=size
        self.depth=0
        self.gates=[]
        self.iteration_number=0
        self.name=name
        if os.path.isdir(self.name+"_Circuits")==False:
            os.makedirs(self.name+"_Circuits")
        if os.path.isdir(self.name+"_Readout")==False:
            os.makedirs(self.name+"_Readout")


    def Log_Parameters(self): #store the information about parameters
        file_name=self.name+"_"+"log.txt"
        write_file=open(file_name,'a')
        for key in param_table:
            write_file.write(str(self.iteration_number)+':parameter:'+key+':'+str(param_table[key])+"\n")
        write_file.close()

    def Log_Functions(self,function,value):#store the information about the experiment results
        file_name=self.name+"_"+"log.txt"
        write_file=open(file_name,'a')
        write_file.write(str(self.iteration_number)+':function:'+function+':'+str(value)+"\n")
        write_file.close()
    def Log_Time(self):#store the information about the experiment results
        file_name=self.name+"_"+"log.txt"
        write_file=open(file_name,'a')
        write_file.write(str(self.iteration_number)+':time:'+str(time.time()-start)+"\n")
        write_file.close()


    def Load_Parameters(self,num_iterations): #load the optimization (probably not useful for OPTaaS)
        file_name=self.name+"_"+"log.txt"
        read_file=open(file_name,"r")
        while True:
            temp=read_file.readline().split(":")
            if len(temp)==0:
                break
            if len(temp[0])==0:
                break
            current_index=int(temp[0])
            if current_index==num_iterations:
                self.iteration_number=num_iterations
                if temp[1]=="parameter":
                    param_table[temp[2]]=float(temp[3])

        read_file.close()
        assert (self.iteration_number==num_iterations)


    def Initialize_Optimization(self,num_probe,**kwarg):
        self.iteration_number=0
        parameters=To_OPTaaS_Params()
        constraints=[]
        self.task = client.create_task(
            title=self.name,
            parameters=parameters,
            constraints=constraints,
            goal=Goal.min,  # or Goal.min as appropriate
            min_known_score=0.001  # optional
            )
        print("Task_ID:"+self.task.id+" started") #log the starting time and ID of current task
        write_file=open("OPTaaS_Operation_Log.txt",'a')
        write_file.write("task ID: "+self.task.id+"  started at  "+str(datetime.datetime.now())[0:10]+"-"+str(datetime.datetime.now())[11:13]+str(datetime.datetime.now())[14:16]+"\n")
        write_file.close()

        if "load_task" in kwarg:
            assert ("number_of_circuits" in kwarg), "specify number_of_circuits to reload"
            old_task=client.get_task(kwarg["load_task"])
            old_results=old_task.get_results(limit=kwarg["number_of_circuits"])
            for result in old_results:
                self.task.add_user_defined_configuration(values=result.configuration.values,score=result.score)
            self.iteration_number=int(np.ceil(kwarg["number_of_circuits"]/num_probe))


        configurations=self.task.generate_configurations(num_probe)
        file_name=self.name+"_Circuits/circuit_"+str("{:03d}").format(self.iteration_number)+".txt"
        write_file=open(file_name,'w')

        for i in range(num_probe):
            From_OPTaaS_Params(configurations[i])
            Regulate_Params()
            write_file.write(self.GatesLab_Sequence()+'\n')

        write_file.close()

        print("Waiting for results of "+file_name)

        return configurations,file_name



    def Update_Result(self,old_configurations,cost_function):
        temp_result=[]
        prefix=self.name+"_Readout/"
        file_list=sorted(os.listdir(prefix),key=lambda x: int(x[x.find("_Line_")+6:x.find(".txt")]))
        temp_state=Quantum_State(self.size)
        num_probe=len(file_list)
        assert(num_probe==len(old_configurations)),"check the result files!"
        assert(int(file_list[0][file_list[0].find("_Line_")+6:file_list[0].find(".txt")])==1), "check the result files!"
        assert(int(file_list[num_probe-1][file_list[num_probe-1].find("_Line_")+6:file_list[num_probe-1].find(".txt")])==num_probe), "check the result files!"



        for index in range(num_probe): #process experiment result, and organize them into the "Result" format
            temp_state.Import(prefix+file_list[index])
            temp_score=cost_function(temp_state)
            self.Log_Functions("probe_"+str(index),temp_score)
            temp_result.append(Result(old_configurations[index],temp_score))


        configurations=self.task.record_results(temp_result) #record the score to get the new set of configurations


        best_result=self.task.get_results(1,best_first=True)[0]#Log the best configuration so far
        From_OPTaaS_Params(best_result.configuration)
        self.Log_Parameters()
        self.Log_Functions("Best Score",best_result.score)
        print("best scroe up to now:",best_result.score)
        self.Log_Time()


        self.iteration_number+=1


        file_name=self.name+"_Circuits/circuit_"+str("{:03d}").format(self.iteration_number)+".txt" # generate circuit file for new configuration
        write_file=open(file_name,'w')

        for i in range(num_probe):
            From_OPTaaS_Params(configurations[i])
            Regulate_Params()
            write_file.write(self.GatesLab_Sequence()+'\n')

        write_file.close()
        print("Waiting for results of "+file_name)

        return configurations,file_name

    def Print_Best_Circuit(self,id_string=""):
        if id_string=="":
            best_result=self.task.get_results(1,best_first=True)[0]#Log the best configuration so far
        else:
            old_task=client.get_task(id_string)
            best_result=old_task.get_results(1,best_first=True)[0]
        From_OPTaaS_Params(best_result.configuration)
        print(self.GatesLab_Sequence())
        return self.GatesLab_Sequence()


    def End_Optimization(self):
        write_file=open("OPTaaS_Operation_Log.txt",'a')
        write_file.write("task ID: "+self.task.id+"  ended at  "+str(datetime.datetime.now())[0:10]+"-"+str(datetime.datetime.now())[11:13]+str(datetime.datetime.now())[14:16]+"\n")
        self.task.complete()
        print("Task_ID:"+self.task.id+" completed")
        write_file.close()

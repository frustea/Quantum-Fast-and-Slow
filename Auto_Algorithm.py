from Core_Definition import *
import os
import datetime
from SPAM import *

class Auto_Algorithm:
    def __init__(self,size):
        self.algorithm_list=[]
        self.size=size

    def Load_Algorithm_List(self,filename):
        self.algorithm_list=[]
        read_file=open(filename,"r")
        while True:
            temp=read_file.readline()
            if len(temp)==0:
                break
            self.algorithm_list.append(temp.rstrip("\n"))
        read_file.close()
    
    def Run(self,path,sample_number=2000):
        temp=Quantum_Circuit(self.size,"whatever")
        for i in range(len(self.algorithm_list)):
            temp.Interpret_GatesLab_Sequence(self.algorithm_list[i])
            filename=str(datetime.datetime.now())[0:10]+"-"+str(datetime.datetime.now())[11:13]+str(datetime.datetime.now())[14:16]+"_Line_"+str("{0:03d}").format(i+1)+".txt"
            temp.Emulate(path+"/"+filename,sample_number)
      
            
        
            
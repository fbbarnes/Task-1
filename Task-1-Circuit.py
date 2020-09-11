#!/usr/bin/env python
# coding: utf-8

# In[62]:



# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.tools.visualization import circuit_drawer
# Loading your IBM Q account(s)
provider = IBMQ.load_account()


# In[67]:


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit.tools.visualization import plot_bloch_multivector

simulator = Aer.get_backend('statevector_simulator')

qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

def EvenBlock(theta):

    circuit.rz(theta[0], qreg_q[0])
    circuit.rz(theta[1], qreg_q[1])
    circuit.rz(theta[2], qreg_q[2])
    circuit.rz(theta[3], qreg_q[3])
    circuit.cz(qreg_q[0], qreg_q[1])
    circuit.cz(qreg_q[0], qreg_q[2])
    circuit.cz(qreg_q[0], qreg_q[3])
    circuit.cz(qreg_q[1], qreg_q[2])
    circuit.cz(qreg_q[1], qreg_q[3])
    circuit.cz(qreg_q[2], qreg_q[3])
    circuit.barrier()
    
def OddBlock(theta):

    circuit.rx(theta[0], qreg_q[0])
    circuit.rx(theta[1], qreg_q[1])
    circuit.rx(theta[2], qreg_q[2])
    circuit.rx(theta[3], qreg_q[3])
    circuit.barrier()

    
def Layer(Layer_params):
    Odd_params = Layer_params[0]
    Even_params = Layer_params[1]
    
    OddBlock(Odd_params)
    EvenBlock(Even_params)
    
def Circuit(Circuit_params):
    for layer_params in Circuit_params:
        Layer(layer_params)

    


theta = np.array([pi/4,0,pi,0.6])
Layer_params = np.stack((theta,theta*2), axis = 0)
Circuit_params = np.stack((Layer_params, Layer_params*2), axis =0)
print(Circuit_params)

#OddBlock(theta)
#EvenBlock(theta*2)

Circuit(Circuit_params)

result = execute(circuit, backend = simulator).result()

statevector = result.get_statevector()

probs = np.absolute(statevector)

# Probabilities for measuring both qubits
circuit.draw()
circuit.draw()
print(statevector)
print(probs)


# In[ ]:





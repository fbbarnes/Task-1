#!/usr/bin/env python
# coding: utf-8



# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
#from qiskit.tools.jupyter import *
#from qiskit.visualization import *
from qiskit.tools.visualization import circuit_drawer
# Loading your IBM Q account(s)
#IBMQ.save_account('8067a84497c769a00d40ce170debac2c1337737eae8b4452ec069158476f75a136475f809fd5db20e72d3662a7937bb8d9a845817dd486b03db4b8dcc1319560')
provider = IBMQ.load_account()



from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import random_statevector, Statevector
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit.tools.visualization import plot_bloch_multivector

#set backend
simulator = Aer.get_backend('statevector_simulator')

#set registers
no_qreg = 4
no_creg = no_qreg

qreg_q = QuantumRegister(no_qreg, 'q')
creg_c = ClassicalRegister(no_creg, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

#define constants
pi = np.pi

#function definitions

def Probabilities(vector):

    vector_conj = np.conjugate(vector)
    vector_sq = vector * vector_conj

    return vector_sq

def Normalise(vector):
    
    vector_sq_sum = np.sum(Probabilities(vector))
    vector_normalised = vector/(np.sqrt(vector_sq_sum))

    return vector_normalised

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

    


#create random target state phi
phi = np.random.rand(2**no_qreg) + np.random.rand(2**no_qreg) * 1j
#normalise phi
phi = Normalise(phi)
#check phi is normalised
#print(np.sum(Probabilities(phi)))


#initialise circuit parameters
#set number of layers
L = 2

#randomise circuit parameters
circuit_params_init = 2*pi * np.random.rand(L, 2, 4)




'''
#example parameters
theta = np.array([pi/4,0,pi,0.6])
Layer_params = np.stack((theta,theta*2), axis = 0)
Circuit_params = np.stack((Layer_params, Layer_params*2), axis =0)
print(Circuit_params)
print(np.shape(Circuit_params))

'''

#make circuit
Circuit(circuit_params_init)


#results
result = execute(circuit, backend = simulator).result()

statevector = result.get_statevector()

probs = np.absolute(statevector)

difference = statevector - phi
distance = np.sum(Probabilities(difference))



# Outputs
print('result ', statevector)
print(np.sum(Probabilities(statevector)))
print('target ', phi)
print(np.sum(Probabilities(phi)))
print('difference', difference)
print(np.sum(Probabilities(difference)))
print('distance', distance)
print(circuit)
circuit.draw(output='mpl')
plt.show()





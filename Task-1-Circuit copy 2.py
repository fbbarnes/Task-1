#!/usr/bin/env python
# coding: utf-8

#pylint: disable=no-member


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



from qiskit.aqua.components.optimizers import AQGD

#set backend
simulator = Aer.get_backend('statevector_simulator')

#set registers
#no_qreg = 4
#no_creg = no_qreg

#qreg_q = QuantumRegister(4, 'q')
#creg_c = ClassicalRegister(4, 'c')
#circuit = QuantumCircuit(qreg_q, creg_c)


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


def EvenBlock(theta,circ):

    circ.rz(theta[0], 0)
    circ.rz(theta[1], 1)
    circ.rz(theta[2], 2)
    circ.rz(theta[3], 3)
    circ.cz(0, 1)
    circ.cz(0, 2)
    circ.cz(0, 3)
    circ.cz(1, 2)
    circ.cz(1, 3)
    circ.cz(2, 3)
    circ.barrier()

    return circ


def OddBlock(theta,circ):

    circ.rx(theta[0], 0)
    circ.rx(theta[1], 1)
    circ.rx(theta[2], 2)
    circ.rx(theta[3], 3)
    circ.barrier()

    return circ


def Layer(Layer_params,circ):
    Odd_params = Layer_params[0]
    Even_params = Layer_params[1]
    
    OddBlock(Odd_params,circ)
    EvenBlock(Even_params,circ)

    return circ


def Circuit(Circuit_params):
    circ = QuantumCircuit(4,4)
    for layer_params in Circuit_params:
        Layer(layer_params,circ)

    return circ




    

#def Loop(circuit):

def Loop(params):
    #randomise circuit parameters
    #print(np.shape(params))
    no_layers = int(params[len(params)-1])
    params = np.delete(params, len(params)-1)
    params = np.reshape(params, (no_layers,2,4))
    #print(np.shape(params))
    
    #print("New circuit being constructed...")

    #circuit =QuantumCircuit(qreg_q,creg_c)


    ''' 
    #example parameters
    theta = np.array([pi/4,0,pi,0.6])
    Layer_params = np.stack((theta,theta*2), axis = 0)
    Circuit_params = np.stack((Layer_params, Layer_params*2), axis =0)
    print(Circuit_params)
    print(np.shape(Circuit_params))

    '''

    #make circuit

    circuit = Circuit(params)


    #results
    result = execute(circuit, backend = simulator).result()

    #reinitialise circuit

    

    statevector = result.get_statevector()

    #print("Target: ",phi)

    difference = statevector - phi
    distance = np.sum(Probabilities(difference))
    #print("Distance:", distance)



    # Outputs
    #print('result ', statevector)
    #print(np.sum(Probabilities(statevector)))
    #print('target ', phi)
    #print(np.sum(Probabilities(phi)))
    #print('difference', difference)
    #print(np.sum(Probabilities(difference)))
    #print('distance', distance)
    #print(circuit)
    #circuit.draw(output='mpl')
    #plt.show()
    circuit =QuantumCircuit(4,4)

    return distance


def UpdateParameters(params, gamma, no_layers):

    grad_Loop = np.zeros(np.size(params))
    new_params = np.zeros(np.shape(params))

    deriv_params = params.flatten()
    deriv_params = np.append(deriv_params, no_layers)

    for j in range(0, np.size(grad_Loop)):
        grad_Loop[j] = AQGD().deriv(j, deriv_params, Loop)

    grad_Loop = np.reshape(grad_Loop, np.shape(new_params))

    new_params = params - gamma * grad_Loop

    return new_params, grad_Loop


def OptimiseParameters(no_loops,no_layers, params_init, gamma):

    distances = np.zeros(no_loops)

    parameters = np.zeros((no_loops,no_layers,2,4))

    grads_Loop  = np.zeros((no_loops-1,no_layers,2,4))



    loops_done =0
    

    for i in range(0,no_loops):
        print("Loop ", i)
        
        if i == 0:
            print("gamma", gamma)
            parameters[i] = params_init
            distances[i] = Loop(np.append(parameters[i], no_layers))
            print("Distance: ", distances[i])
            print("shape parameters[i]:", np.shape(params_init))
            print("parameters", parameters)
            loops_done += 1


        elif i == 1:


            print("gamma: ", gamma)
            parameters[i], grads_Loop[i-1] = UpdateParameters(parameters[i-1], gamma, no_layers)
            distances[i] = Loop(np.append(parameters[i], no_layers))
            print("Distance: ", distances[i])
            loops_done += 1



        else:

            delta_parameters = (parameters[i-1] - parameters[i-2]).flatten()

            delta_grads = (grads_Loop[i-1] - grads_Loop[i-2]).flatten()


            #gamma =   np.dot(delta_parameters, delta_grads) / (np.dot(delta_grads,delta_grads))
            gamma =   np.dot(delta_parameters, delta_parameters) / (np.dot(delta_parameters,delta_grads))

            print("gamma ", gamma)
    
            parameters[i], grads_Loop[i-1] = UpdateParameters(parameters[i-1], gamma, no_layers)
            distances[i] = Loop(np.append(parameters[i], no_layers))
            print("Distance: ", distances[i])
            

            if distances[i] > distances[i-1]:
                print("Converged after loop ", i)
                loops_done += 1

            else:
                loops_done += 1


    return parameters[0:loops_done], distances[0:loops_done]



def GetDistances(layers, no_loops, params_init, gamma):

    all_distances = np.zeros((len(layers), no_loops))


    for i in range(0, len(layers)):

        print("Calculating distances for ", layers[i], " layers")


        parameters, all_distances[i] = OptimiseParameters(no_loops, layers[i], params_init,gamma)


    print("all_distances ", all_distances)
    min_distances = np.amin(all_distances, axis = 1)

    return min_distances



#create random target state phi
phi = np.random.uniform(-1, 1, 2**4) + np.random.uniform(-1, 1, 2**4) * 1j
#normalise phi
phi = Normalise(phi)
#check phi is normalised
#print(np.sum(Probabilities(phi)))


#initialise circuit parameters
#set number of layers
#set number of iterations (loops)
NO_LOOPS = 10
#set initial parameters

#print(np.shape(initial_parameters))
#create array for layers to consider
LAYERS_ARRAY = np.arange(1,10)

INITIAL_PARAMETERS = 2 * pi * np.random.rand(LAYERS_ARRAY[0], 2, 4)

#set learning rate
GAMMA = 0.1


MINIMUM_DISTANCES = GetDistances(LAYERS_ARRAY,NO_LOOPS,INITIAL_PARAMETERS,GAMMA)

print(MINIMUM_DISTANCES)
print(LAYERS_ARRAY)

#plt.plot(LAYERS_ARRAY, MINIMUM_DISTANCES)
#plt.show()

title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
              'verticalalignment':'center'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'20'}

plt.plot(LAYERS_ARRAY, MINIMUM_DISTANCES)


plt.xlabel("Layers", **axis_font)
plt.ylabel(r"$\epsilon$", **axis_font)
#plt.title("Minimum distance after 100 iterations and learning rate of 0.1", **title_font)


plt.savefig("test.png")
plt.show()




'''
parameters, distances = OptimiseParameters(no_loops, no_layers, initial_parameters,gamma)



final_circuit = Circuit(parameters[(parameters.shape[0]-1)])

final_result = execute(final_circuit, backend = simulator).result()

final_statevector = final_result.get_statevector()


print("Distances: ", distances)
print("Final parameters: ", parameters[(parameters.shape[0]-1)])
print("Target: ", phi)
print("Solution: ", final_statevector)

#phi_bloch = plot_bloch_multivector(phi, title='phi')
#final_bloch = plot_bloch_multivector(final_statevector, title='final_statevector')
#plt.show()
'''

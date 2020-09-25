#!/usr/bin/env python
# coding: utf-8

#pylint: disable=no-member


# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import random_statevector, Statevector
from qiskit.tools.visualization import plot_bloch_multivector
from qiskit.aqua.components.optimizers import AQGD
# Import numpy
import numpy as np
# Import matplotlib
import matplotlib
import matplotlib.pyplot as plt


#Set Qiskit backen to statevector
simulator = Aer.get_backend('statevector_simulator')


#define constants
pi = np.pi


#function definitions

#calclate state probabilities from a statevector
def Probabilities(vector):

    vector_conj = np.conjugate(vector)
    vector_sq = vector * vector_conj

    return vector_sq


#normalise a statevector
def Normalise(vector):
    
    vector_sq_sum = np.sum(Probabilities(vector))
    vector_normalised = vector/(np.sqrt(vector_sq_sum))

    return vector_normalised


#function to create paramaterised gates for even block in circuit
def EvenBlock(theta,circ):

    #make Rz rotations for each qubit
    circ.rz(theta[0], 0)
    circ.rz(theta[1], 1)
    circ.rz(theta[2], 2)
    circ.rz(theta[3], 3)
    #make CZ gates cascade
    circ.cz(0, 1)
    circ.cz(0, 2)
    circ.cz(0, 3)
    circ.cz(1, 2)
    circ.cz(1, 3)
    circ.cz(2, 3)
    circ.barrier()

    return circ


#function to create paramaterised gates for odd block in circuit
def OddBlock(theta,circ):

    #create Rx rotations for each qubit
    circ.rx(theta[0], 0)
    circ.rx(theta[1], 1)
    circ.rx(theta[2], 2)
    circ.rx(theta[3], 3)
    circ.barrier()

    return circ


#function to create paramaterised layers consisting of both blocks in circuit
def Layer(Layer_params,circ):
    Odd_params = Layer_params[0]
    Even_params = Layer_params[1]
    
    OddBlock(Odd_params,circ)
    EvenBlock(Even_params,circ)

    return circ


#function to create paramaterised circuit consisting of multiple layers according to dimensions of parameters
def Circuit(Circuit_params):
    circ = QuantumCircuit(4,4)
    for layer_params in Circuit_params:
        Layer(layer_params,circ)

    return circ


#function to loop through in order to execute circuit with given an array
# of the variational parameters with the number of layers (L) as the last element and 
# calculate distances (|| |(ψ(θ))> - |(φ(θ))> ||) given parameters
# note: uses global PHI variable for target state vector
def Loop(params):
    #get L from input array
    no_layers = int(params[len(params)-1])
    #remove L from input array to leave rotation angles only
    params = np.delete(params, len(params)-1)
    params = np.reshape(params, (no_layers,2,4))

    #make circuit
    circuit = Circuit(params)

    #calculate results as a statevector array
    result = execute(circuit, backend = simulator).result()
    statevector = result.get_statevector()

    #calulate ε
    difference = statevector - PHI
    distance = np.sum(Probabilities(difference))

    #reinitialise circuit
    circuit =QuantumCircuit(4,4)

    return distance

#function to find gradient of Loop() at the parameters given for a given no of layers
def GradLoop(params, no_layers):

    #create new array for gradient
    grad = np.zeros(np.size(params))

    #flatten parameters so it is compatible with AQGD.deriv()
    deriv_params = params.flatten()
    #add L to end of circuit parameter array
    deriv_params = np.append(deriv_params, no_layers)

    #for each parameter, caclulate the partial deriviative of cost function
    # calculated by Loop() 
    for j in range(0, np.size(grad)):
        grad[j] = AQGD().deriv(j, deriv_params, Loop)

    #reshape the array into the same as the parameters array
    grad = np.reshape(grad, np.shape(params))

    return grad



#function to optimise the parameters such as to minimise the distance between the
# target state and circuit state using gradient descent with the Barzilai-Borwei method. 
#also updates user in console of progress by displaying intermediary results
def OptimiseParameters(no_loops,no_layers, params_init, gamma_init):

    #create new arrays for calculated quantities
    distances = np.zeros(no_loops)
    parameters = np.zeros((no_loops,no_layers,2,4))
    grads_Loop  = np.zeros((no_loops-1,no_layers,2,4))
    gammas = np.zeros(no_loops-1)
    #initialise number of iterations
    loops_done =0
    
    #iterate through loops
    for i in range(0,no_loops):
        #tell user which loop is starting
        print("Loop ", i)
        
        #for initial parameters calculate gradient and starting distance
        if i == 0:
            #use initial parameters for first calculation of cost function
            parameters[i] = params_init
            distances[i] = Loop(np.append(parameters[i], no_layers))
            #set initial value for learning rate
            gammas[i] = gamma_init
            #calculate gradient for first set of parameters (to be used in next loop)
            grads_Loop[i] = GradLoop(params_init, no_layers)

            loops_done += 1

        #for second term in gradient descent use initial learning rate and update distance
        elif i ==1:
            #update parameters according to gradient descent
            parameters[i] = parameters[i-1] - gammas[i-1] * grads_Loop[i-1]
            #calculate new ε
            distances[i] = Loop(np.append(parameters[i], no_layers))
            #Tell user current distance and learning rate
            print("Distance: ", distances[i])
            print("gamma ", gammas[i-1])

            loops_done += 1

        #for all other terms calculate gradient at previous parameters, update the learning rate using Barzilai-Borwei(BB) method
        else:
            #calculate gradient
            grads_Loop[i-1] = GradLoop(parameters[i-1], no_layers)

            #intermediate calculations for BB method
            grads_difference = (grads_Loop[i-1] - grads_Loop[i-2]).flatten()
            params_difference = (parameters[i-1] - parameters[i-2]).flatten()
            #update learning rate according to BB
            gammas[i-1] = np.abs(np.dot(params_difference,grads_difference) / np.dot(grads_difference, grads_difference))

            #update parameters using gradient desecent
            parameters[i] = parameters[i-1] - gammas[i-1] * grads_Loop[i-1]
            #calculate new ε
            distances[i] = Loop(np.append(parameters[i], no_layers))
            #console output
            print("Distance: ", distances[i])
            print("gamma ", gammas[i-1])


            #check if result is converging by comparing percentage change of last two terms to 0.1%
            if (  0 > ((distances[i-1] - distances[i])/ distances[i]) > -0.001):
                #find out what loop we're in
                converged_loop = i
                #tell user that we've converged
                print("Converged after loop ", converged_loop)
                loops_done += 1
                #stop calculating ε for this layer
                break


            loops_done += 1

    #return all parameters and distances calculate before converging
    return parameters[0:loops_done], distances[0:loops_done]

#function to create random parameters for the gates in the circuit for a given number of layers
def RandomParameters(no_layers):

    rand_params = 2 * pi * np.random.rand(no_layers, 2, 4)

    return rand_params


#function to calculate minimum distances achieved through optimisation
#for a given an array for the  layers to use, no of iterations (loops) and initial learning rate
#also outputs all intermediate distances into console
def GetDistances(layers, no_loops, gamma):

    #create array for all distances calculated
    all_distances = []

    #loop through each circuit of L layers
    for i in range(0, len(layers)):
        #tell user which layer we're on
        print("Calculating distances for ", layers[i], " layers")

        #set initial parameters randomly
        params_init = RandomParameters(layers[i])

        #optimise parameters to get parameters and distances
        parameters, distances = OptimiseParameters(no_loops, layers[i], params_init,gamma)

        #add to array of all distances
        all_distances.append(distances)


    print("all_distances ", all_distances)
    #find minimum value of distance for each circuit with L layers
    min_distances = [np.amin(a) for a in all_distances]

    return min_distances



#GLOBAL VARIABLES
#create random target state PHI = |(φ(θ))>
PHI = np.random.uniform(-1, 1, 2**4) + np.random.uniform(-1, 1, 2**4) * 1j
#normalise phi
PHI = Normalise(PHI)

#set number of iterations (loops)
NO_LOOPS = 200
#create array for layers to consider
LAYERS_ARRAY = np.arange(1,10)

#set initial learning rate
GAMMA = 0.1

#Calculate the minimum distances (ε) for the desired layers,
# no. of iterations and inital learning rate
MINIMUM_DISTANCES = GetDistances(LAYERS_ARRAY,NO_LOOPS,GAMMA)




#OUTPUT RESULTS
print(MINIMUM_DISTANCES)
print(LAYERS_ARRAY)

#CREATE PLOT
#set fonts
title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
              'verticalalignment':'center'} 
axis_font = {'fontname':'Arial', 'size':'20'}

#plot ε against no of layers
plt.plot(LAYERS_ARRAY, MINIMUM_DISTANCES)

#label axis
plt.xlabel("Layers", **axis_font)
plt.ylabel(r"$\epsilon$", **axis_font)

#save figure
plt.savefig("Plot.png")
#plot in console
plt.show()
# Task-1

## 0 Introduction
This repository contains the work carried out as part of the screening process for the Quantum Open Source Foundation mentorship programme. It aims to explore the problem set out in Task 1. [Code.](https://github.com/fbbarnes/Task-1/blob/master/Task-1-Circuit.py)

## 1 Task
The task is described as follows:

##### 1.1
Implement, on a quantum simulator of your choice, the following 4 qubit state |(ψ)>:

![circuit diagram](https://github.com/fbbarnes/Task-1/blob/master/figures/circuit-diagram.png)

Where the number of layers, denoted with L, has to be considered as a parameter. We call ¨Layer¨ the combination of 1 yellow + 1 green block, so, for example, U<sub>1</sub> + U<sub>2</sub> is a layer. The odd/even variational blocks are given by:

![block diagram](https://github.com/fbbarnes/Task-1/blob/master/figures/blocks-diagram.png)

The angles θ<sub>i</sub>, are variational parameters, lying in the interval (0, 2), initialized at random. Double qubit gates are CZ gates.

##### 1.2
Report with a plot, as a function of the number of layers, L, the minimum distance:

ε = min<sub>θ</sub> || |(ψ(θ))> - |(φ(θ))> ||,

where |(φ(θ))> is a randomly generated vector on 4 qubits and the norm || | v> ||, of a state | v>, simply denotes the square root of the sum of the modulus square of the components of |v >. The right set of parameters i,n can be found via any method of choice (e.g. grid-search or gradient descent)

## 2 Approach

##### 2.1
The 4 qubit state was implemented using Qiskit Terra. The circuit was constructed using qiskit.circuit. The functions used to create the gates in the circuit were implemented to allow parameterisation of the R<sub>z</sub> and R<sub>x</sub> gate angles as well as the number of layers in the circuit. The resulting circuit was simulated using StateVectorSimulator from Qiskit Aer. The resulting statevector was taken as a representation of |(ψ)> in the computational basis. 

##### 2.2
The target state,  |(φ(θ))>, was represented by generating an array of 2<sup>4</sup> random numbers in the complex plane with modulus ≤ 1 and normalising. After initialsing the variational parameters randomly, the circuit was simulated and the resulting statevector used to calculate  || |(ψ(θ))> - |(φ(θ))> || as described above. 

In order to find the correct paramaters to find the minimum distance, ε, || |(ψ(θ))> - |(φ(θ))> || was treated as a cost function to be minimised using gradient descent. The iterative process can be described by the equation:

θ<sub>n</sub> = θ<sub>n-1</sub> - γ<sub>n-1</sub> ∇ε(θ<sub>n-1</sub>),

where γ is the learning rate and ∇ε is the gradient of the cost function with respect to the variational parameters, θ. 

The gradient was calculated using the deriv method from Qiskit Aqua's analytic quantum gradient descent optimiser class (AQGD). This method is based on parameter-shift rules in which the analytic derivative for a quantum gate can be calculated simply by finding the difference between the resulting states for two sets of parameters (NB: this is distinct from finite-differences methods). 

The learning rate, γ, was varied using the Barzilai-Borwei method. This method iterates γ using the following relation:

γ<sub>n</sub> = |Δθ.Δε| / |Δε|<sup>2</sup>,

where Δθ = θ<sub>n</sub> - θ<sub>n-1</sub> and Δε = ∇ε(θ<sub>n</sub>) - ∇ε(θ<sub>n-1</sub>). γ<sub>0</sub> was taken as 0.1. 

For each new set of variational parameters, the distance was calculated and stored. For each number of layers, L, gradient descent would be iterated n times and ε taken as the minimum value of the distances. This was repeated for each L and ε was plotted against L using matplotlib. 

##### 2.3
The code for the task was written in python and is in the file [Task-1-Circuit.py](https://github.com/fbbarnes/Task-1/blob/master/Task-1-Circuit.py). The Qiskit, NumPy, and Matplotlib libraries are required to run it. Running it will produce a graph like that below.

## 3 Results
A plot of the results for 1 ≤ L ≤ 6 and n = 100 is presented below. 

![plot of results](https://github.com/fbbarnes/Task-1/blob/master/figures/example-plot.png)

Through trial and error it was found that roughly less than 100 iterations were sufficient to converge on a value for ε for each L < 10. However, for values of L ≤ 3, the algorithim tended to find local minima as repeated runs gave different values for ε. This suggests that the Barzilai-Borwei method for learning parameter adjustment may be improved upon. Nevertheless, the Barzilai-Borwei method was more successful than use of just a constant learning rate. 

For L ≥ 5, the algorithm produced values for ε << 10<sup>-25</sup>, ie approximately zero.  This is consistent with the findings in https://arxiv.org/pdf/quant-ph/0602174.pdf, which proposes a general quantum circuit pattern that can be used to implement any n-qubit quantum gate. The circuit shares a similar structure to the one used in this task by having layers of one-qubit gates and cascades of controlled Pauli gates. It is shown that n+1 layers are required, consistent with the results here. 

## 4 Further questions to explore
##### 4.1 Is it possible to use fewer/different gates and still successfully minimise the distance?
https://arXiv:quant-ph/0407010  states 2<sup>n+1</sup> − 2 one-qubit rotations and ⌈1/4 * (2<sup>n+1</sup> − 3n − 2)⌉ CNOT gates is the lower bound for the number of qubits to produce an arbitary state with n qubits. 
##### 4.2 Is it possible to get faster results using a different optimisation algorithm?
Even with only a handful of qubits and layers the calculations are intensive on a classical computer. Would different classical simulations and algorithms help us to explore larger quantum circuits? 
##### 4.3 What is the optimal number of layers?
The results here show that increasing the depth of the circuit for a fixed number of qubits will not help one get a more precise answer after a certain point. But will using a smaller/larger number of layers require fewer/more iterations to find |φ>? What is the associated computational cost? This question particuarly important went considering NISQs where noise and coherence times play an important role. 

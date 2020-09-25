# Task-1

## 0 Introduction
This repository contains the work carried out as part of the screening process for the Quantum Open Source Foundation mentorship programme. It aims to explore the problem set out in Task 1.

## 1 Task
The task is described as follows:

##### 1.1
Implement, on a quantum simulator of your choice, the following 4 qubit state |(ψ)>:

Where the number of layers, denoted with L, has to be considered as a parameter. We call ¨Layer¨ the combination of 1 yellow + 1 green block, so, for example, U<sub>1</sub> + U<sub>2</sub> is a layer. The odd/even variational blocks are given by:

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

## 3 Results
A plot of the results for 1 ≤ L ≤ 9 and n = 100 is presented below. 

Through trial and error it was found that roughly less than 100 iterations were sufficient to converge on a value for ε for each L < 10. However, for values of L ≤ 3, the algorithim tended to find local minima as repeated runs gave different values for ε. This suggests that the Barzilai-Borwei method for learning parameter adjustment may be improved upon. Nevertheless, the Barzilai-Borwei method was more successful than use of just a constant learning rate. For L ≥ 4, the algorithm consistently produced values for ε << 10<sup>-10<\sup>, ie approximately zero. This corresponds to 24 CZ gates, 16 R<sub>x</sub>, and 16 R<sub>z</sub> gates. 

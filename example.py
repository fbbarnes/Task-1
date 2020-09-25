
import numpy as np

@profile
def Probabilities(vector):

    vector_conj = np.conjugate(vector)
    vector_sq = vector * vector_conj

    return vector_sq


@profile
def Normalise(vector):
    
    vector_sq_sum = np.sum(Probabilities(vector))
    vector_normalised = vector/(np.sqrt(vector_sq_sum))

    return vector_normalised

test_vector = np.random.rand(1, 3)
probabilites = Probabilities(test_vector)
normalised = Normalise(test_vector)
normalised_probabilites = Probabilities(normalised)

print("test_vector \n", test_vector)
print("Probabilities \n", probabilites)
print("Normalise \n", normalised)
print("Normalised probabilites \n", normalised_probabilites)

print("Sum \n", np.sum(normalised_probabilites))
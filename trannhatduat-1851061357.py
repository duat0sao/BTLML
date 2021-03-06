"""
    Trần Nhật Duât - 1851061357 - 60TH4
    
"""


import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        np.random.seed(1)
        
        self.synaptic_weights = 2 * np.random.random((6, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        for iteration in range(training_iterations):
            output = self.think(training_inputs)

            error = training_outputs - output
            
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    #training data 10 mẫu đầu và 5 mẫu cuối
    training_inputs = np.array([[4.7,3.2,1.3,0.2,1,0,],
                                [4.7,3.1,1.5,0.2,1,0,],
                                [5.7,3.9,1.7,0.4,1,0,],
                                [4.7,3.4,1.4,0.3,1,0,],
                                [5.7,3.4,1.5,0.2,1,0,],
                                [4.7,2.9,1.4,0.2,1,0,],
                                [4.7,3.1,1.5,0.1,1,0,],
                                [5.7,3.7,1.5,0.2,1,0,],
                                [4.7,3.4,1.6,0.2,1,0,],
                                [4.7,3.0,1.4,0.1,1,0,],
                                [4.7,3.0,1.1,0.1,1,0,],
                                [5.7,4.0,1.2,0.2,1,0,],
                                [5.7,4.4,1.5,0.4,1,0,],
                                [5.7,3.9,1.3,0.4,1,0,],
                                [5.7,3.5,1.4,0.3,1,0,],
                                [5.7,3.8,1.7,0.3,1,0,],
                                [5.7,3.8,1.5,0.3,1,0,],
                                [5.7,3.4,1.7,0.2,1,0,],
                                [5.7,3.7,1.5,0.4,1,0,],
                                [4.7,3.6,1.0,0.2,1,0,],
                                [5.7,3.3,1.7,0.5,1,0,],
                                [4.7,3.4,1.9,0.2,1,0,],
                                [5.7,3.0,1.6,0.2,1,0,],
                                [5.7,3.4,1.6,0.4,1,0,],
                                [5.7,3.5,1.5,0.2,1,0,],
                                [5.7,3.4,1.4,0.2,1,0,],
                                [4.7,3.2,1.6,0.2,1,0,],
                                [4.7,3.1,1.6,0.2,1,0,],
                                [5.7,3.4,1.5,0.4,1,0,],
                                [5.7,4.1,1.5,0.1,1,0,],
                                [5.7,4.2,1.4,0.2,1,0,],
                                [4.7,3.1,1.5,0.2,1,0,],
                                [5.7,3.2,1.2,0.2,1,0,],
                                [5.7,3.5,1.3,0.2,1,0,],
                                [4.7,3.6,1.4,0.1,1,0,],
                                [4.7,3.0,1.3,0.2,1,0,],
                                [5.7,3.4,1.5,0.2,1,0,],
                                [7.7,3.2,4.7,1.4,0,1,],
                                [6.7,3.2,4.5,1.5,0,1,],
                                [6.7,3.1,4.9,1.5,0,1,],
                                [5.7,2.3,4.0,1.3,0,1,],
                                [6.7,2.8,4.6,1.5,0,1,],
                                [5.7,2.8,4.5,1.3,0,1,],
                                [6.7,3.3,4.7,1.6,0,1,],
                                [4.7,2.4,3.3,1.0,0,1,],
                                [6.7,2.9,4.6,1.3,0,1,],
                                [5.7,2.7,3.9,1.4,0,1,],
                                [5.7,2.0,3.5,1.0,0,1,],
                                [5.7,3.0,4.2,1.5,0,1,],
                                [6.7,2.2,4.0,1.0,0,1,],
                                [6.7,2.9,4.7,1.4,0,1,],
                                [5.7,2.9,3.6,1.3,0,1,],
                                [6.7,3.1,4.4,1.4,0,1,],
                                [5.7,3.0,4.5,1.5,0,1,],
                                [5.7,2.7,4.1,1.0,0,1,],
                                [6.7,2.2,4.5,1.5,0,1,],
                                [5.7,2.5,3.9,1.1,0,1,],
                                [5.7,3.2,4.8,1.8,0,1,],
                                [6.7,2.8,4.0,1.3,0,1,],
                                [6.7,2.5,4.9,1.5,0,1,],
                                [6.7,2.8,4.7,1.2,0,1,],
                                [6.7,2.9,4.3,1.3,0,1,],
                                [6.7,3.0,4.4,1.4,0,1,],
                                [6.7,2.8,4.8,1.4,0,1,],
                                [6.7,3.0,5.0,1.7,0,1,],
                                [6.7,2.9,4.5,1.5,0,1,],
                                [5.7,2.6,3.5,1.0,0,1,],
                                [5.7,2.4,3.8,1.1,0,1,],
                                [5.7,2.4,3.7,1.0,0,1,],
                                [5.7,2.7,3.9,1.2,0,1,],
                                [6.7,2.7,5.1,1.6,0,1,],
                                [5.7,3.0,4.5,1.5,0,1,],
                                [6.7,3.4,4.5,1.6,0,1,],
                                [6.7,3.1,4.7,1.5,0,1,],
                                [6.7,2.3,4.4,1.3,0,1,],
                                [5.7,3.0,4.1,1.3,0,1,],
                                [5.7,2.5,4.0,1.3,0,1,],
                                [6.7,3.3,6.0,2.5,0,0,],
                                [5.7,2.7,5.1,1.9,0,0,],
                                [7.7,3.0,5.9,2.1,0,0,],
                                [6.7,2.9,5.6,1.8,0,0,],
                                [6.7,3.0,5.8,2.2,0,0,],
                                [7.7,3.0,6.6,2.1,0,0,],
                                [4.7,2.5,4.5,1.7,0,0,],
                                [7.7,2.9,6.3,1.8,0,0,],
                                [6.7,2.5,5.8,1.8,0,0,],
                                [7.7,3.6,6.1,2.5,0,0,],
                                [6.7,3.2,5.1,2.0,0,0,],
                                [5.7,2.5,5.0,2.0,0,0,],
                                [5.7,2.8,5.1,2.4,0,0,],
                                [6.7,3.2,5.3,2.3,0,0,],
                                [6.7,3.0,5.5,1.8,0,0,],
                                [7.7,3.8,6.7,2.2,0,0,],
                                [7.7,2.6,6.9,2.3,0,0,],
                                [6.7,3.2,5.7,2.3,0,0,],
                                [5.7,2.8,4.9,2.0,0,0,],
                                [7.7,2.8,6.7,2.0,0,0,],
                                [6.7,2.7,4.9,1.8,0,0,],
                                [6.7,3.3,5.7,2.1,0,0,],
                                [7.7,3.2,6.0,1.8,0,0,],
                                [6.7,2.8,4.8,1.8,0,0,],
                                [6.7,3.0,4.9,1.8,0,0,],
                                [6.7,2.8,5.6,2.1,0,0,],
                                [7.7,3.0,5.8,1.6,0,0,],
                                [7.7,2.8,6.1,1.9,0,0,],
                                [7.7,3.8,6.4,2.0,0,0,],
                                [6.7,2.8,5.6,2.2,0,0,],
                                [6.7,2.8,5.1,1.5,0,0,],
                                [6.7,2.6,5.6,1.4,0,0,],
                                [7.7,3.0,6.1,2.3,0,0,],
                                [6.7,3.4,5.6,2.4,0,0,],
                                [6.7,3.1,5.5,1.8,0,0,],
                                [6.7,3.0,4.8,1.8,0,0,],
                                [6.7,3.1,5.4,2.1,0,0,]])

    training_outputs = np.array([[0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,1,1,1,
                                  1,1,1,1,1,1,1,1,1,1,
                                  1,1,1,1,1,1,1,1,1,1,
                                  1,1,1,1,1,1,1,1,1,1,
                                  1,1,1,1]]).T

    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    #Mẫu TEST: 5 mẫu đầu, 5 mẫu cuối
    test_inputs =     np.array([[5.7,3.5,1.3,0.3,1,0],
                                [4.7,2.3,1.3,0.3,1,0],
                                [4.7,3.2,1.3,0.2,1,0],
                                [5.7,3.5,1.6,0.6,1,0],
                                [5.7,3.8,1.9,0.4,1,0],
                                [4.7,3.0,1.4,0.3,1,0],
                                [5.7,3.8,1.6,0.2,1,0],
                                [4.7,3.2,1.4,0.2,1,0],
                                [5.7,3.7,1.5,0.2,1,0],
                                [5.7,3.3,1.4,0.2,1,0],
                                [5.7,2.6,4.4,1.2,0,1],
                                [6.7,3.0,4.6,1.4,0,1],
                                [5.7,2.6,4.0,1.2,0,1],
                                [5.7,2.7,4.2,1.3,0,1],
                                [5.7,3.0,4.2,1.2,0,1],
                                [5.7,2.9,4.2,1.3,0,1],
                                [6.7,2.9,4.3,1.3,0,1],
                                [5.7,2.5,3.0,1.1,0,1],
                                [5.7,2.8,4.1,1.3,0,1],
                                [6.7,3.1,5.6,2.4,0,0],
                                [6.7,3.1,5.1,2.3,0,0],
                                [5.7,2.7,5.1,1.9,0,0],
                                [6.7,3.3,5.7,2.5,0,0],
                                [6.7,3.0,5.2,2.3,0,0],
                                [6.7,2.5,5.0,1.9,0,0],
                                [6.7,3.0,5.2,2.0,0,0],
                                [6.7,3.4,5.4,2.3,0,0],
                                [5.7,3.0,5.1,1.8,0,0]])

    test_outputs =      np.array([[0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,
                                  1,1,1,1,1,1,1,1,1]]).T

    print("New Output data: ")
    print(neural_network.think(test_inputs))


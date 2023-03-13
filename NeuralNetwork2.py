import numpy as np
# import nnfs
from nnfs.datasets import spiral_data
import torch
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""Dette dokument indeholder den kode der bruges i opgaven, men også noget kode som ikke bliver brugt.
Hvis det lyster kan man ændre mængden af lag og neuroner på linje 424. indsæt tal i networkShape og husk at adskil
med komma. Tallet bestemmer hvor mange neuroner der er og mængden af tal i listen bestemmer mængden af lag.
Husk at sætte en mængde af aktiveringsfunktioner der svare til mængden af lag - 1, fordi input laget ikke har en
aktiveringsfunktion. Jeg anbefaler at bruge softmax til det sidste lag da det datasæt jeg har brugt er en
klassifiserings opgave"""

"""I opgaven bruger jeg det eksempel som er indlæst nu, så hvis du vil se hvordan det virker så lad være med
at ændre noget"""

"""Jeg har brugt en video serie fra Sentdex på Youtube til at lave meget af forwardprobagation koden og 
initialiseringen af netværkert. Jeg har brugt del 1, 2, 3, 4, 5 og 6. Meget af koden har jeg sidenhen modificeret
til at passe til det jeg skulle bruge det til så det kan godt være det ikke ligner på en prik det som han
viser i videoen:
https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=1
Jeg har også brugt et dokument der forklare matematikken til backprobagation til at skrive koden til
backprobagation som bruges til optimeringen:
https://www.3blue1brown.com/lessons/backpropagation-calculus
Koden for backprobagation og de andre aktiverings funktioner end ReLu og softmax har jeg skrevet med udgangspunkt 
i andre dokumenter"""

"""Koden er skrevet af mig med hjælp fra de kilder jeg har nævnt her. Der har også været andre kilder til de
andre aktiveringsfunktioner, men da de aktiveringsfunktioner ikke er blevet brugt til eksemplet i opgaven har
jeg ikke citeret dem
Nogle af kommentarene er på engelsk. Det er fordi jeg er tosproget og er mere vandt til at bruge engelsk når
jeg koder fordi jeg har lært kodning fra engelske kilder online."""


# nnfs.init()

# class for layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, multiplier, zero=False):
        # change 1 to 0.1 if it isn't used with sin function
        if zero == False:
            self.weights = multiplier * np.random.randn(n_inputs,
                                              n_neurons)  # creates already transposed value matrix for weights
        else:
            self.weights = np.zeros((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
        self.activation = np.zeros(n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases




# class for network
class Neural_Network:
    def __init__(self, shape, activation_types):
        self.shape = shape



        self.activations = []
        self.layers = []  # builds network
        for i in range(0, len(shape) - 1):
            self.activations.append(activation_types[i]())
            if i == 0:
                zero = True
            else:
                zero = False

            if self.activations[i].type == "sin":
                self.layers.append(Layer_Dense(shape[i], shape[i + 1], 1.5, zero))
            else:
                self.layers.append(Layer_Dense(shape[i], shape[i + 1], 0.1, zero))

        self.cum_change = np.zeros((784, self.layers[0].biases.size))

    def forward(self, X):  # forwards data through the network
        for l in range(0, len(self.layers)):
            self.layers[l].forward(X)

            if l == len(self.layers) - 1:
                self.activations[l].forward(self.layers[l].output)  # for the last layer
                self.layers[l].activation = self.activations[l].output
                return self.activations[l].output
            else:
                self.activations[l].forward(self.layers[l].output)
                self.layers[l].activation = self.activations[l].output
                X = self.activations[l].output  # for all layers that aren't the last

    def back_propagate(self, X, y, loss_function, learning_rate=0.1):


        loss = loss_function
        layer_size = self.layers[len(self.layers) - 1].biases.size  # finds size of last layer

        # onehot encode y
        onehot_encode_y = np.zeros((len(y), layer_size))
        for i in range(0, len(y)):
            onehot_encode_y.T[y[i], i] = 1


        """
        find deriative of cost with relation to weight and bias on layer one
        use chain rule to find deriative of cost with relation to earlier layers
        Create gradient vector
        multiply gradient vector with learningrate and add to current weight values
        """

        # create array for gradient vector for weights
        dC_dw_gradient = []
        # create array for gradient vector for biases
        dC_db_gradient = []
        for l in range(0, len(self.shape)-1):
            dC_dw_gradient.append(np.zeros((self.shape[l], self.shape[l+1], len(y))))
            dC_db_gradient.append(np.zeros((self.shape[l+1], len(y))))

        # for the last layer
        l = len(self.shape) - 2

        a = 1  # debugging purposes.

        # derivative of neuron value before activation with respect to weights
        if len(self.layers) != 1:
            dz_dw = np.array(self.layers[l - 1].activation)
        else:
            dz_dw = np.array(X)

        # deriative of activation with respect to neuron output

        da_dz = np.array(self.activations[l].deriv(self.layers[l].output))

        # deriative of cost with respect to activation
        dc_da = np.array(loss.deriv(self.layers[l].activation, onehot_encode_y))


        for k in range(0, dC_dw_gradient[l].shape[1]):
            dC_db_gradient[l][k] = da_dz.T[k] * dc_da.T[k]
            for j in range(0, dC_dw_gradient[l].shape[0]):
                # chain rule means that we can multiply all the deriatives to get dc_dw
                dC_dw_gradient[l][j, k] = dz_dw.T[j] * da_dz.T[k] * dc_da.T[k]


        # for all other layers than the last
        for o in range(0, len(self.shape)-2):
            l = len(self.shape)-3 - o

            a = 1  # debugging purposes.

            # deriative of layer after this layer (l+1)'s output with respect this layers acitvation
            dz_da = np.array(self.layers[l+1].weights)


            # deriative of cost with respect to activation
            new_dc_da = np.zeros((len(y), self.shape[l+1]))
            for j in range(0, self.shape[l+1]):
                # notation might be a bit confusing, but i am using the un-updated numbers to get the numbers from
                # previous layers
                """ there is a bug where dc_da changes when this loop iterates, 
                even though none of the code changes it"""
                #a = np.multiply(dz_da[j], da_dz, dc_da)
                new_dc_da.T[j] = np.sum((dz_da[j] * da_dz * dc_da), axis=1)

            dc_da = new_dc_da

            # deriative of activation with respect to neuron output
            da_dz = np.array(self.activations[l].deriv(self.layers[l].output))



            if l != 0:
                dz_dw = np.array(self.layers[l - 1].activation)
            else:
                dz_dw = np.array(X)

            # using chain rule deriative of cost with respect to weight

            for k in range(0, dC_dw_gradient[l].shape[1]):
                dC_db_gradient[l][k] = da_dz.T[k] * dc_da.T[k]
                for j in range(0, dC_dw_gradient[l].shape[0]):
                    a = dz_dw.T[j] * da_dz.T[k] * dc_da.T[k]
                    dC_dw_gradient[l][j, k] = dz_dw.T[j] * da_dz.T[k] * dc_da.T[k]


        """the following section has nothing to do with the optimization, but in debugging mode it gives insight
        into what makes the network fire and what the network is looking for"""


        debug = 0
        if debug == 1:
            l = -1
            dz_da = np.array(self.layers[l+1].weights)


            # deriative of cost with respect to activation
            new_dc_da = np.zeros((len(y), self.shape[l+1]))
            for j in range(0, self.shape[l+1]):
                # notation might be a bit confusing, but i am using the un-updated numbers to get the numbers from
                # previous layers
                """ there is a bug where dc_da changes when this loop iterates, 
                even though none of the code changes it"""
                #a = np.multiply(dz_da[j], da_dz, dc_da)
                new_dc_da.T[j] = np.sum((dz_da[j] * da_dz * dc_da), axis=1)

            Debug_dc_da = new_dc_da

            Debug_input_matrix = np.zeros((28, 28))
            for o in range(0, 64):
                for i in range(0, 28):
                    Debug_input_matrix[i] = Debug_dc_da[o][i * 28:(i + 1) * 28] * 100

                    for q in range(0, 28):
                        if X[o][i*28 + q] == 0:
                            Debug_input_matrix[i][q] = 0

        # gradient vectors have been found, now i subtract them from the weights and biases
        for l in range(0, len(self.shape)-1):
            Debug_weights = -np.mean(dC_dw_gradient[l], axis=2) * learning_rate
            Debug_weights_matrix = np.zeros((28, 28))
            Debug_bias = -np.mean(dC_db_gradient[l], axis=1) * learning_rate
            self.layers[l].weights -= np.mean(dC_dw_gradient[l], axis=2) * learning_rate
            self.layers[l].biases -= np.mean(dC_db_gradient[l], axis=1) * learning_rate

            """den følgende par klumper kode gør heller ikke noget men hvis man har adgang til et debugging
            værktøj kan man bruge det til at få noget indsigt i hvordan netværket lærer. Det synes jeg selv
            er lidt sjovt. Bare sæt q til at være 1 enten mens man er i debugging mode eller bare her i koden,
            hver dog opmærksom på at det gør koden en del langsommerer."""

            if l== 0:
                self.cum_change -= np.mean(dC_dw_gradient[l], axis=2) * learning_rate * learning_rate

            q = 0
            if q == 1:
                vector_length = 0
                for i in range(0, Debug_weights.shape[0]):
                    for o in range(0, Debug_weights.shape[1]):
                        vector_length += Debug_weights[i, o]**2
                vector_length = np.sqrt(vector_length)

            q = 0
            if q == 1:
                Debug_change_weights_matrix = np.zeros((28, 28))
                if l == 0:
                    for o in range(0, 10):
                        for i in range(0, 28):
                            Debug_weights_matrix[i] = self.layers[l].weights.T[o][i*28:(i+1)*28]
                            Debug_change_weights_matrix[i] = self.cum_change.T[o][i*28:(i+1)*28]




class Activation_binary_step:
    def __init__(self):
        self.output = None
        self.type = "binary_step"


    def forward(self, inputs):
        """funktionen svare ikke helt til signumfunktionen som der bliver forklaret om i opgaven, men det er
        fordi min kode ligger op til at netværket skal outputte 0 istedet for -1. Dette giver bedre mening
        i forhold til hvis koden skal bruges til større netværker og gør derfor koden mere fleksibel.
        Det burde dog være samme resultat læringsmæssigt da der ikke forventes noget -1 fra perceptronen"""
        self.output = inputs > 0


    def deriv(self, inputs):
        """koden jeg skriver her svare ikke til den afledte signum funktion da den afledte signum funkion
        er lig med 0. Koden her skal istedet gøre at den funktion der bruges til at rette vægte og bias
        er ækvivalent med den funktion som der står i opgaven. I opgaven indeholder den formel der bruges til at
        opdatere vægtene ikke den differentierede funktion, derfor får jeg denne funktion til at outputte
        1-taller for at der ikke bliver tage højde for den differentierede signum funktion.
        Denne del af funktionen er der kun så aktiveringsfunktionen kan virke med resten af koden som er skrevet
        med formål til at virke i mulit-lags neurale netværk
        Grunden til at jeg dividere med 2 er at cost-funktionens afledte funktion er 2(y-a), så jeg dividere
        med to her for at gøre det ækvivalent til det der står i opgaven"""

        return np.ones(inputs.shape)/2





class Activation_sin:
    def __init__(self):
        self.output = None
        self.type = "sin"

    def forward(self, inputs):
        self.output = np.sin(inputs)

    def deriv(self, inputs):
        return np.cos(inputs)



# rectified linear activation function
class Activation_ReLU:  # ReLU activation function, cuts off below 0
    def __init__(self):
        self.output = None
        self.type = "ReLu"

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def deriv(self, inputs):
        return inputs > 0  # if inputs is over 0 returns 1, else returns 0

class Activation_Leaky_ReLU:  # ReLU activation function, cuts off below 0
    def __init__(self):
        self.output = None
        self.type = "Leaky_ReLu"

    def forward(self, inputs):
        self.output = np.maximum(0.02*inputs, inputs)

    def deriv(self, inputs):
        is_positive = inputs > 0
        return np.maximum(0.02, is_positive)


# Softmax, squezes activations between 0 and 1
class Activation_Softmax:
    def __init__(self):
        self.change = 0.0000001
        self.type = "softmax"
        self.output = None

    def forward(self, inputs, return_value = False):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # on default the function doesn't return the values, it just stores it as self.output.
        # if return_value is specified it will return the output rather than storing it
        if return_value == False:
            self.output = probabilities
        else:
            return probabilities

    def deriv(self, inputs):

        """i softmax påviker et output alle de andre fordi det kommer ud i procent som i alt giver 100
        Jeg er i tvivl om hvorvidt jeg skal finde hvor meget en ændring i input påvirker alle outputs eller
        om jeg kun skal finde hvor meget en ændring i input påvirker activerinen af den neuron som inputet
        kommer fra"""

        deriv_inputs = np.zeros(inputs.T.shape)
        for x in range(0, inputs.T.shape[0]):
            change_vector = np.zeros(inputs.T.shape[0])
            change_vector[x] = self.change
            # opstiller en differential kvotient: (a(z+h) - a(z))/h. h er change vector og er tæt på 0
            deriv_inputs[x] = ((self.forward(inputs + change_vector, True) - self.forward(inputs, True))/ self.change).T[x]

        return deriv_inputs.T




# calculates loss
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # data_loss = np.mean(sample_losses)
        # print("average loss: ", np.mean(sample_losses))
        return sample_losses


# forward method varies, for chess use loss = difference**2
class Loss_Squared_Differences(Loss):
    def forward(self, y_pred, y_true):

        if y_true.ndim == 1:
            onehot_encode_y = np.zeros((len(y), y_pred.shape[1]))

            for i in range(0, len(y)):
                onehot_encode_y.T[y[i], i] = 1
            y_true = onehot_encode_y



        return np.sum((y_pred - y_true)**2, axis=1)

    def deriv(self, y_pred, y_true):
        a = 2*(y_pred - y_true)
        return 2*(y_pred - y_true)

def accuracy(y_pred, y_true):
    right = 0

    for i in range(len(y_true)):
        prediction = 0
        highest = 0
        for o in range(len(y_pred[0])):
            if y_pred[i, o] > highest:
                highest = y_pred[i, o]
                prediction = o
        if y_true[i] == prediction:
            right += 1


    return 100/(len(y_true)) * right




Samplesize = 64
samples_trained = 0


a = samples_trained
y = y_train[a:a+Samplesize]
X = []
for i in range(0, Samplesize):
    X.append([])
    X[i] = x_train[a + i].flatten()/255

samples_trained += Samplesize



#for o in range(0, len(X)-1):
    #X[o] = X[o].flatten()

#X, y = spiral_data(samples=Samplesize, classes=3)

print(X, y)

networkShape = [784, 10]

activations = [Activation_binary_step]
network = Neural_Network(networkShape, activations)

network_output = network.forward(X)

print(network_output)

loss_function = Loss_Squared_Differences()

print("Loss1:", loss_function.calculate(network.forward(X), y))

# optimization

epochs = 1
total_epochs = 1
total_samples = 60000
timer = 0

learning_rate = 0.1

level = 0

while epochs != 0:

    if epochs == 1:
        debug = 1

    if Samplesize + samples_trained < 60000:
        pass
    else:
        samples_trained = 0
        epochs -= 1
        print("Epoch: ", total_epochs - epochs)


    a = samples_trained
    y = y_train[a:a + Samplesize]
    X = []
    for i in range(0, Samplesize):
        X.append([])
        X[i] = x_train[a + i].flatten()/255

    samples_trained += Samplesize



    #X, y = spiral_data(samples=Samplesize, classes=3)


    network_output = network.forward(X)
    network.back_propagate(X, y, loss_function, learning_rate)

    if timer == 5:
        print("percent done: ", samples_trained / total_samples * 100)
        print("average loss: ", np.mean(loss_function.calculate(network_output, y)))
        print("average accuracy: ", accuracy(network_output, y), "%")
        if np.mean(loss_function.calculate(network_output, y)) < 0.5:
            if level == 0:
                learning_rate = 0.1  # changes learning rate when cost is lower to avoid
            level = 1

        if np.mean(loss_function.calculate(network_output, y)) < 0.3:
            learning_rate = 0.1  # changes learning rate when cost is lower to avoid
        timer = 0

    timer += 1

    if i == total_samples - 2:
        debug = 0



Samplesize = 10000

y = y_test[0:Samplesize]
X = []
for i in range(0, Samplesize):
    X.append([])
    X[i] = x_test[i].flatten()/255


network_output = network.forward(X)


print("test loss: ", np.mean(loss_function.calculate(network_output, y)))
print("test accuracy: ", accuracy(network_output, y), "%")














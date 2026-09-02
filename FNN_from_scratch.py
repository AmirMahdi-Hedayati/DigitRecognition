import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

batch_size = 64
learning_rate = 0.05
epochs = 120

class Activation:
    def __init__(self):
        self.input = None

    def forward(self , x):
        raise NotImplementedError
    
    def backward(self , x):
        raise NotImplementedError


class Relu(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0 , x)
    
    def backward(self, previous_grad):
        return previous_grad * (self.input > 0)
    
class Linear(Activation):
    def forward(self, x):
        self.input = x
        return x

    def backward(self, previous_grad):
        return previous_grad
    

class Soft_max_activation(Activation):
    def forward(self, x):
        self.input = x 
        logits = self.input - np.max(self.input , axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        self.output = exp_logits / np.sum(exp_logits , axis = 0 , keepdims=True)
        return self.output
    
    def backward(self, x):
        pass
    

class Layer:

    def __init__(self , output_size , previous_layer , activation):
        self.activation = activation
        self.output_size = output_size
        self.previous_layer = previous_layer
        self.next_layer = None
        self.input_size = previous_layer.output_size
        self.weights = np.random.randn(self.output_size , self.input_size) * np.sqrt(2.0 / self.input_size)
        self.bias = np.zeros((self.output_size , 1))
        self.grad_bias = None
        self.grad_weights = None
        self.input = None 
        self.output = None
        self.grad_input = None

    def forward(self):

        self.input = self.previous_layer.output
        z = self.weights @ self.input + self.bias
        self.output = self.activation.forward(z)
        return self.output

    def backward(self , next_grad):
        
        z_grads = self.activation.backward(next_grad)
        self.grad_bias = np.sum(z_grads , axis = 1 , keepdims = True)
        self.grad_weights = z_grads @ self.input.T
        self.grad_input = self.weights.T @ z_grads
        return self.grad_input
    
    def update(self):
        self.weights = self.weights - learning_rate * self.grad_weights
        self.bias = self.bias - learning_rate * self.grad_bias


class Softmax(Layer):

    def __init__(self, previous_layer):
        self.activation = Soft_max_activation()
        self.previous_layer = previous_layer
        self.input_size = previous_layer.output_size
        self.input = None 
        self.output = None
        self.grad_input = None

    def forward(self , labels):
        self.input = self.previous_layer.output
        self.output = self.activation.forward(self.input)
        l = -1 * labels * np.log(self.output + 1e-12)
        self.loss = np.mean(np.sum(l, axis=0))
        self.labels = labels
        return self.loss

    def backward(self):
        self.grad_input = (self.output - self.labels) / self.output.shape[1]
        return self.grad_input
    
    def update(self):
        pass
    

class Inputlayer(Layer):
    def __init__(self , output_size):
        self.output_size = output_size
        self.output = None

    def set_output(self , output):
        self.output = output



class fully_connected_network:
    def __init__(self):
        self.layers = []
        self.losses = []
        self.input = Inputlayer(64)
        self.layers.append(Layer(40 , self.input , Relu()))
        self.layers.append(Layer(24 , self.layers[-1] , Relu()))
        self.layers.append(Layer(15 , self.layers[-1] , Relu()))
        self.layers.append(Layer(10 , self.layers[-1] , Linear()))
        self.layers.append(Softmax(self.layers[-1]))
    
    def forward(self , batch , one_hot_labels):
        self.input.set_output(batch)
        for layer in self.layers[:-1]:
            layer.forward()
        loss = self.layers[-1].forward(one_hot_labels)
        return loss

    def backward(self):
        last_grad = self.layers[-1].backward()

        for layer in reversed(self.layers[:-1]):
            last_grad = layer.backward(last_grad)

    def optimize(self):
        for layer in self.layers:
            layer.update()

    def train(self , train_data , train_labels , size):
        for _ in range(epochs):
        
            indices = np.arange(train_data.shape[0])
            np.random.shuffle(indices)
            train_data = train_data[indices]
            train_labels = train_labels[indices]

            for start in range(0 , size , batch_size):
                batch = train_data[start:start+batch_size].T
                labels = train_labels[start:start+batch_size].T
                loss = self.forward(batch , labels)
                self.backward()
                self.optimize()

            self.losses.append(loss)
            print(loss)

    def test(self , test_data , test_labels):
        return self.forward(test_data.T , test_labels.T)
    


def main():

    data, labels = load_digits(return_X_y=True)
    data = data.astype(np.float64) / 16.0
    labels = np.eye(10)[labels]
    
    train_data, test_data , train_labels, test_labels = train_test_split( data, labels, test_size=0.3, random_state=42)

    print(f"train size : {train_labels.size} \n test size : {test_labels.size}")


    network = fully_connected_network()
    network.train(train_data, train_labels , train_data.shape[0])


    loss = network.test(test_data , test_labels)
    print(f"average test loss is : {loss}")


if __name__ == "__main__":
    main()
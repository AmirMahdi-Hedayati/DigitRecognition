import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view

batch_size = 64
learning_rate = 0.05
epochs = 120

class Activation:
    def __init__(self):
        self.input = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

class Relu(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

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
        logits = self.input - np.max(self.input, axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        self.output = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        return self.output

    def backward(self, previous_grad):
        raise NotImplementedError

class Layer:
    def __init__(self, output_shape, previous_layer, weight_shape, activation, fan_in, tensor_dot_axis):
        self.activation = activation
        self.output_shape = output_shape
        self.output_size = output_shape[0]
        self.previous_layer = previous_layer
        self.next_layer = None
        self.input_size = previous_layer.output_size
        self.weights = np.random.standard_normal(weight_shape) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros((output_shape[0],) + (1,) * (len(output_shape) - 1))
        self.tensor_dot_axis = tensor_dot_axis
        self.grad_bias = None
        self.grad_weights = None
        self.tensor_dot_input = None
        self.output = None
        self.grad_input = None

    def forward(self, input):
        self.tensor_dot_input = input
        z = np.tensordot(self.weights, input, axes=(self.tensor_dot_axis["weights"], self.tensor_dot_axis["input"])) + self.bias
        self.output = self.activation.forward(z)
        return self.output

    def backward(self, next_grad):
        z_grads = self.activation.backward(next_grad)
        self.grad_bias = np.sum(z_grads, axis=tuple(self.tensor_dot_axis["output"]), keepdims=True)
        self.grad_weights = np.tensordot(z_grads, self.tensor_dot_input, axes=(self.tensor_dot_axis["output"], self.tensor_dot_axis["input_t"]))
        self.grad_input = np.tensordot(self.weights, z_grads, axes=(self.tensor_dot_axis["weights_t"], self.tensor_dot_axis["output_t"]))
        return self.grad_input

    def update(self):
        self.weights = self.weights - learning_rate * self.grad_weights
        self.bias = self.bias - learning_rate * self.grad_bias

class FullyConnected(Layer):
    def __init__(self, output_size, previous_layer, activation):
        axes = {
            "weights": [1], "input": [0],
            "output": [1], "input_t": [1],
            "weights_t": [0], "output_t": [0]
        }
        super().__init__(
            (output_size, None), previous_layer,
            (output_size, previous_layer.output_size),
            activation, previous_layer.output_size, axes
        )

class Softmax(Layer):
    def __init__(self, previous_layer):
        self.activation = Soft_max_activation()
        self.previous_layer = previous_layer
        self.input_shape = previous_layer.output_shape
        self.input = None
        self.output = None
        self.grad_input = None

    def forward(self, labels):
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

class Flatting(Layer):
    def __init__(self, previous_layer, axis, transpose):
        self.previous_layer = previous_layer
        self.axis = axis
        self.transpose = transpose
        self.inverse_transpose = np.argsort(tuple(transpose))
        c, _, h, w = previous_layer.output_shape
        self.output_size = c * h * w
        self.output_shape = (self.output_size, None)

    def forward(self):
        self.output = self.previous_layer.output.transpose(
            tuple(self.transpose)).reshape(-1, self.previous_layer.output.shape[self.axis])
        return self.output

    def backward(self, next_grad):
        new_shape = list(self.previous_layer.output.shape)
        new_shape.append(new_shape.pop(self.axis))
        return next_grad.reshape(tuple(new_shape)).transpose(self.inverse_transpose)

    def update(self):
        pass

class Convolution(Layer):
    def __init__(self, output_shape, previous_layer, weight_shape, activation, fan_in, tensor_dot_axis, stride):
        super().__init__(output_shape, previous_layer, weight_shape, activation, fan_in, tensor_dot_axis)
        self.stride = stride
        self.kernel_size = weight_shape[-1]

    def forward(self, input):
        self.input_shape = input.shape
        windows = sliding_window_view(input, (self.kernel_size, self.kernel_size), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride]
        patches = windows.transpose(0, 4, 5, 1, 2, 3)
        return super().forward(patches)

    def backward(self, next_grad):
        patch_grads = super().backward(next_grad)
        self.grad_input = np.zeros(self.input_shape)
        h_out, w_out = patch_grads.shape[-2:]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.grad_input[:, :, i:i + h_out*self.stride:self.stride,
                                j:j + w_out*self.stride:self.stride] += patch_grads[:, i, j]
        return self.grad_input

class Inputlayer(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape
        self.output_size = output_shape[0]
        self.output = None

    def set_output(self, output):
        self.output = output

class fully_connected_network:
    def __init__(self):
        self.layers = []
        self.losses = []
        self.input = Inputlayer((1, None, 8, 8))

        conv_axes = {
            "weights": [1, 2, 3], "input": [0, 1, 2],
            "output": [1, 2, 3], "input_t": [3, 4, 5],
            "weights_t": [0], "output_t": [0]
        }

        self.layers.append(Convolution((20, None, 5, 5), self.input, (20 , 1, 4, 4), Relu(), 16, conv_axes, 1))
        self.layers.append(Convolution((8, None, 2, 2), self.layers[-1], (8, 20, 4, 4), Relu(), 320 , conv_axes, 1))
        self.layers.append(Flatting(self.layers[-1], 1, (0, 2, 3, 1)))
        self.layers.append(FullyConnected(10, self.layers[-1], Linear()))
        self.layers.append(Softmax(self.layers[-1]))

    def forward(self, batch, one_hot_labels):
        self.input.set_output(batch)
        for layer in self.layers[:-1]:
            if isinstance(layer, Flatting):
                layer.forward()
            else:
                layer.forward(layer.previous_layer.output)
        loss = self.layers[-1].forward(one_hot_labels)
        return loss

    def backward(self):
        last_grad = self.layers[-1].backward()
        for layer in reversed(self.layers[:-1]):
            last_grad = layer.backward(last_grad)

    def optimize(self):
        for layer in self.layers:
            layer.update()

    def train(self, train_data, train_labels, size):
        for epoch in range(epochs):
            indices = np.arange(train_data.shape[0])
            np.random.shuffle(indices)
            train_data = train_data[indices]
            train_labels = train_labels[indices]
            total_loss = 0.0
            for start in range(0, size, batch_size):
                batch = train_data[start:start + batch_size].reshape(-1, 8, 8)[None, :, :, :]
                labels = train_labels[start:start + batch_size].T
                loss = self.forward(batch, labels)
                self.backward()
                self.optimize()
                total_loss += loss * batch.shape[1]
            self.losses.append(total_loss / size)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"epoch {epoch + 1}: loss {self.losses[-1]:.4f}")

    def test(self, test_data, test_labels):
        batch = test_data.reshape(-1, 8, 8)[None, :, :, :]
        return self.forward(batch, test_labels.T)

def main():
    np.random.seed(42)
    data, labels = load_digits(return_X_y=True)
    data = data.astype(np.float64) / 16.0
    labels = np.eye(10)[labels]
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.3, random_state=42
    )
    print(f"train size: {len(train_labels)}, test size: {len(test_labels)}")
    network = fully_connected_network()
    network.train(train_data, train_labels, train_data.shape[0])
    loss = network.test(test_data, test_labels)
    accuracy = np.mean(
        np.argmax(network.layers[-1].output, axis=0) ==
        np.argmax(test_labels, axis=1)
    )
    print(f"average test loss is: {loss:.4f}, accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
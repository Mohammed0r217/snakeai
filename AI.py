import torch
import torch.nn as nn
import numpy as np
from pygame.draw import circle, line


class Model(nn.Module):

    def __init__(self, shape):
        super(Model, self).__init__()

        self.layers = []
        self.connections = []
        self.model, self.loss_fn, self.optimizer = self.build(shape)

    def build(self, shape):
        model = nn.Sequential()

        model.add_module('input', nn.Linear(shape[0], shape[1], bias=False))
        # model.add_module('relu1', nn.ReLU())

        for i in range(len(shape) - 3):
            model.add_module(f'hidden{i+1}', nn.Linear(shape[i+1], shape[i+2], bias=False))
            # model.add_module(f'relu{i}', nn.ReLU())

        model.add_module('output', nn.Linear(shape[-2], shape[-1], bias=False))
        model.add_module('softmax', nn.Softmax(dim=1))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        for layer in model:
            if isinstance(layer, nn.Linear):
                weights = layer.weight.tolist()
                self.connections.append(weights)

        return model, loss_fn, optimizer

    def forward(self, x):
        output = None
        self.layers = [x.tolist()[0]]

        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.Softmax) or isinstance(module, nn.Linear):
                if output is None:
                    output = module(x)
                else:
                    output = module(output)

                if isinstance(module, nn.Linear) or isinstance(module, nn.Softmax):
                    self.layers.append(output.tolist()[0])

        del self.layers[-2]

        return output

    def predict(self, input_):
        x = torch.tensor(input_)
        with torch.no_grad():
            output = self.forward(x)

        return output

    def fit(self, x, y, batch_size, epochs):
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                inputs = np.array(x[i:i + batch_size])
                targets = np.array(y[i:i + batch_size])

                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets).float()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                loss.backward()
                self.optimizer.step()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def render(self, window):
        pass

    def crossover(self, parent1, parent2):
        dict_parent1 = parent1.state_dict()
        dict_parent2 = parent2.state_dict()

        dict_child = dict_parent1.copy()

        for key in dict_parent1.keys():  # key defines the layer and the parameters (weights or biases)

            tensor_parent1 = dict_parent1[key]  # get parameters
            tensor_parent2 = dict_parent2[key]

            # crossover point is in the middle
            index = tensor_parent1.numel() // 2

            # Flatten the tensors and perform crossover
            flat_tensor1 = tensor_parent1.view(-1)
            flat_tensor2 = tensor_parent2.view(-1)

            flat_tensor1[index:], flat_tensor2[index:] = flat_tensor2[index:], flat_tensor1[index:]

            # Reshape the tensors and assign them to the child
            dict_child[key] = flat_tensor1.view(*tensor_parent1.shape)  # * unpachs the tuple

        # Load the child state dict into the current model
        self.load_state_dict(dict_child)


class RLModel(Model):
    def __init__(self, shape):
        super().__init__(shape)

        self.type = 'RL'
        self.input = []
        self.output = []

    def render(self, window):
        pass


class GAModel(Model):
    generation = 1
    shown = True

    def __init__(self, shape):
        super().__init__(shape)

        self.fitness = 0
        self.type = 'GA'

    def calcFitness(self, completion, weighted_avg_path_length):
        self.fitness = 10 * (completion ** 1.7) / (weighted_avg_path_length ** 0.8)
        return self.fitness

    """def mutate(self, mutation_rate):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.numpy()
                biases = layer.bias.data.numpy()

                # Determine how many weights and biases to mutate
                num_weights = weights.size
                num_biases = biases.size

                num_mutations = int(mutation_rate * (num_weights + num_biases))

                # Randomly select weights and biases to mutate
                mutation_indices = np.random.choice(num_weights + num_biases, num_mutations, replace=False)

                # Split mutation indices into weights and biases
                weight_indices = mutation_indices[mutation_indices < num_weights]
                bias_indices = mutation_indices[mutation_indices >= num_weights] - num_weights

                # Apply mutations to weights
                weight_mutations = np.random.normal(loc=0.0, scale=0.5, size=len(weight_indices))
                np.put(weights, weight_indices, weights.take(weight_indices) + weight_mutations)

                # Apply mutations to biases
                bias_mutations = np.random.normal(loc=0.0, scale=0.5, size=len(bias_indices))
                np.put(biases, bias_indices, biases.take(bias_indices) + bias_mutations)

                layer.weight.data.copy_(torch.from_numpy(weights))
                layer.bias.data.copy_(torch.from_numpy(biases))"""

    def mutate(self, mutation_rate):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.numpy()

                # Determine how many weights to mutate
                num_weights = weights.size
                num_mutations = int(mutation_rate * num_weights)

                # Randomly select weights to mutate
                mutation_indices = np.random.choice(num_weights, num_mutations, replace=False)

                # Apply mutations
                mutations = np.random.normal(loc=0.0, scale=0.5, size=num_mutations)
                np.put(weights, mutation_indices, weights.take(mutation_indices) + mutations)

                layer.weight.data.copy_(torch.from_numpy(weights))

                weights = layer.weight.tolist()
                self.connections.append(weights)

    def copy(self, source):
        self.model.load_state_dict(source.model.state_dict())

    def render(self, window):
        if not GAModel.shown:
            return

        r = 14
        y_spacing = 5
        x_spacing = 90
        x = 40
        y = 800 - (2 * r + y_spacing) * max([len(x) for x in self.layers]) / 2

        for layer in range(len(self.layers)):
            x1 = x + x_spacing * layer

            for node in range(len(self.layers[layer])):
                y1 = y - (2 * r + y_spacing) * len(self.layers[layer]) / 2 + (2 * r + y_spacing) * node

                if layer < (len(self.layers) - 1):
                    for i in range(len(self.layers[layer + 1])):
                        x2 = x1 + x_spacing
                        y2 = y - (2 * r + y_spacing) * len(self.layers[layer + 1]) / 2 + (2 * r + y_spacing) * i
                        positive_w = (self.connections[layer][i][node] > 0)
                        line(window, (255 * (not positive_w), 0, 255 * positive_w), (x1, y1), (x2, y2), 2)

                a = self.layers[layer][node]

                if layer == 0:  # input layer
                    if a >= 0:
                        c = (round(255 * abs(1. - a)), 255, round(255 * abs(1. - a)))
                    else:
                        c = (255, round(255 * abs(1. + a)), 255)

                elif layer == len(self.layers) - 1:  # output layer
                    c = (round(255 * abs(1. - a)), 255, round(255 * abs(1. - a)))

                else:
                    highest = max([abs(n) for n in self.layers[layer]])
                    if highest == 0:
                        highest = 0.00001
                    a /= highest
                    if a >= 0:
                        c = (255, 255, round(255 * abs(1. - a)))
                    else:
                        c = (255, 165 + round(90 * abs(1. + a)), round(255 * abs(1. + a)))

                circle(window, c, (x1, y1), radius=r, width=0)

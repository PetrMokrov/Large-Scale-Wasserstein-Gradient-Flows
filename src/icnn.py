import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from copy import copy

from .layers import ConvexQuadratic, View, WeightTransformedLinear

class GradNN(nn.Module):
    def __init__(self, batch_size=1024):
        super(GradNN, self).__init__()        
        self.batch_size = batch_size
    
    def forward(self, input):
        pass
    
    def push(self, input, create_graph=True, retain_graph=True):
        '''
        Pushes input by using the gradient of the network. By default preserves the computational graph.
        # Apply to small batches.
        '''
        if len(input) <= self.batch_size:
            output = autograd.grad(
                outputs=self.forward(input), inputs=input,
                create_graph=create_graph, retain_graph=retain_graph,
                only_inputs=True,
                grad_outputs=torch.ones_like(input[:, :1], requires_grad=False)
            )[0]
            return output
        else:
            output = torch.zeros_like(input, requires_grad=False)
            for j in range(0, input.size(0), self.batch_size):
                output[j: j + self.batch_size] = self.push(
                    input[j:j + self.batch_size],
                     create_graph=create_graph, retain_graph=retain_graph)
            return output
    
    def push_nograd(self, input):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        for i in range(0, len(input), self.batch_size):
            input_batch = input[i:i+self.batch_size]
            output.data[i:i+self.batch_size] = self.push(
                input[i:i+self.batch_size],
                create_graph=False, retain_graph=False
            ).data
        return output
    
    def hessian(self, input):
        gradient = self.push(input)
        hessian = torch.zeros(
            *gradient.size(), self.dim,
            dtype=torch.float32,
            requires_grad=True,
        )

        hessian = torch.cat(
            [
                torch.autograd.grad(
                    outputs=gradient[:, d], inputs=input,
                    create_graph=True, retain_graph=True,
                    only_inputs=True, grad_outputs=torch.ones(input.size()[0]).float().to(input)
                )[0][:, None, :]
                for d in range(self.dim)
            ],
            dim = 1
        )
        return hessian

class LinDenseICNN(GradNN):
    '''
    Fully Connected ICNN which follows the [Makkuva et.al.] article:
    (https://arxiv.org/pdf/1908.10962.pdf)
    '''

    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32], 
        activation=torch.celu,
        strong_convexity=1e-6,
        batch_size=1024,
        device='cuda'):
        raise Exception("Not working yet!")

        super().__init__(batch_size)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.in_dim = in_dim
        self.device = device
        self.strong_convexity = strong_convexity

        _hidden = copy(self.hidden_layer_sizes)
        w_sizes = zip(_hidden[:-1], _hidden[1:])

        self.W_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in w_sizes
        ])

        self.A_layers = nn.ModuleList([
            nn.Linear(self.in_dim, out_dim) 
            for out_dim in _hidden
        ])
        self.final_layer = nn.Linear(self.hidden_layer_sizes[-1], 1, bias=False)
        self.to(self.device)
    
    def forward(self, input):
        z = self.activation(self.A_layers[0](input))
        for a_layer, w_layer in zip(self.A_layers[1:], self.W_layers[:]):
            z = self.activation(a_layer(input) + w_layer(z))
        
        return self.final_layer(z)  + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)    

    def convexify(self):
        for layer in self.W_layers:
            assert isinstance(layer, nn.Linear)
            layer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
    
    def relaxed_convexity_regularization(self):
        regularizer = 0.
        for layer in self.W_layers:
            assert isinstance(layer, nn.Linear)
            regularizer += layer.weight.clamp(max=0.).pow(2).sum()
        regularizer += self.final_layer.weight.clamp(max=0.).pow(2).sum()
        return regularizer

class DenseICNN(GradNN):
    '''Fully Conncted ICNN with input-quadratic skip connections.'''
    def __init__(
        self, dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu',
        strong_convexity=1e-6,
        batch_size=1024,
        conv_layers_w_trf=lambda x: x,
        forse_w_positive=True
    ):
        super(DenseICNN, self).__init__(batch_size)
        
        self.dim = dim
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.conv_layers_w_trf = conv_layers_w_trf
        self.forse_w_positive = forse_w_positive
        
        self.quadratic_layers = nn.ModuleList([
            ConvexQuadratic(dim, out_features, rank=rank, bias=True)
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            WeightTransformedLinear(
                in_features, out_features, bias=False, w_transform=self.conv_layers_w_trf)
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = WeightTransformedLinear(
            hidden_layer_sizes[-1], 1, bias=False, w_transform=self.conv_layers_w_trf)

    def forward(self, input):
        '''Evaluation of the discriminator value. Preserves the computational graph.'''
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def convexify(self):
        if self.forse_w_positive:
            for layer in self.convex_layers:
                if (isinstance(layer, nn.Linear)):
                    layer.weight.data.clamp_(0)
            self.final_layer.weight.data.clamp_(0)
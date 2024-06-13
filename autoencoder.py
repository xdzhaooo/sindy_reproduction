import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FullNetwork(nn.Module):
    def __init__(self, params):
        super(FullNetwork, self).__init__()
        self.input_dim = params['input_dim']
        self.latent_dim = params['latent_dim']
        self.activation = params['activation']
        self.poly_order = params['poly_order']
        self.include_sine = params.get('include_sine', False)
        self.library_dim = params['library_dim']
        self.model_order = params['model_order']

        self.encoder = self.build_autoencoder(params, True)
        self.decoder = self.build_autoencoder(params, False)

        # self.sindy_coefficients = nn.Parameter(torch.randn(self.library_dim, self.latent_dim))
        # self.coefficient_mask = nn.Parameter(torch.ones(self.library_dim, self.latent_dim), requires_grad=False)
        #self.coefficient_mask =params['coefficient_mask']
        self.coefficient_mask = torch.nn.Parameter(torch.ones(self.library_dim, self.latent_dim), requires_grad=False)

                # Initialize sindy_coefficients based on the parameter
        if params['coefficient_initialization'] == 'xavier':
            self.sindy_coefficients = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty(self.library_dim, self.latent_dim))
            )
        elif params['coefficient_initialization'] == 'specified':
            self.sindy_coefficients = torch.nn.Parameter(
                torch.tensor(params['init_coefficients'], dtype=torch.float32)
            )
        elif params['coefficient_initialization'] == 'constant':
            self.sindy_coefficients = torch.nn.Parameter(
                torch.ones(self.library_dim, self.latent_dim)
            )
        elif params['coefficient_initialization'] == 'normal':
            self.sindy_coefficients = torch.nn.Parameter(
                torch.nn.init.normal_(torch.empty(self.library_dim, self.latent_dim))
            )
        else:
            raise ValueError('Invalid coefficient initialization method')

        # Handle sequential thresholding if needed
        self.sequential_thresholding = params.get('sequential_thresholding', False)
        if self.sequential_thresholding:
            print("Sequential thresholding is enabled")
            self.coefficient_mask = torch.nn.Parameter(params['coefficient_mask'], requires_grad=False)


    def build_autoencoder(self, params, is_encoder):
        layers = []
        if is_encoder:
            input_dim = self.input_dim
            output_dim = self.latent_dim
        else:
            input_dim = self.latent_dim
            output_dim = self.input_dim

        widths = params['widths']
        activation = self.get_activation_function(params['activation'])

        if is_encoder:
            for width in widths:
                layers.append(nn.Linear(input_dim, width))
                layers.append(activation)
                input_dim = width
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            for width in reversed(widths):
                layers.append(nn.Linear(input_dim, width))
                layers.append(activation)
                input_dim = width
            layers.append(nn.Linear(input_dim, output_dim))

        return nn.Sequential(*layers)

    def get_activation_function(self, activation):
        if activation == 'linear':
            return nn.Identity()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError('Invalid activation function')

    def sindy_library(self, z):
        library = [torch.ones(z.size(0)).to(z.device)]
        for i in range(self.latent_dim):
            library.append(z[:,i])

        if self.poly_order > 1:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library.append(z[:,i] * z[:,j])

        if self.poly_order > 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    for k in range(j, self.latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k])

        if self.poly_order > 3:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    for k in range(j, self.latent_dim):
                        for p in range(k, self.latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

        if self.poly_order > 4:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    for k in range(j, self.latent_dim):
                        for p in range(k, self.latent_dim):
                            for q in range(p, self.latent_dim):
                                library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

        if self.include_sine:
            for i in range(self.latent_dim):
                library.append(torch.sin(z[:,i]))

        # print(torch.stack(library, dim=1).shape)
        # print("cat")
        # print(torch.cat(library, dim=1).shape)

        return torch.stack(library, dim=1).to(z.device)
        
    def sindy_library_order2(self, z, dz):
        """
        Build the SINDy library for a second-order system in PyTorch.
        """
        z = torch.cat([z, dz], dim=1)
        latent_dim = self.latent_dim*2
        library = [torch.ones(z.size(0)).to(z.device)]

        for i in range(latent_dim):
            library.append(z[:,i])

        if self.poly_order > 1:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    library.append(z[:,i] * z[:,j])

        if self.poly_order > 2:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    for k in range(j,latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k])

        if self.poly_order > 3:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    for k in range(j,latent_dim):
                        for p in range(k,latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

        if self.poly_order > 4:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    for k in range(j,latent_dim):
                        for p in range(k,latent_dim):
                            for q in range(p,latent_dim):
                                library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

        if self.include_sine:
            for i in range(latent_dim):
                library.append(torch.sin(z[:,i]))

        return torch.stack(library, dim=1).to(z.device)

    def forward(self, x, dx, ddx=None):
            # 确保输入张量需要计算梯度
        x.requires_grad_(True)
        dx.requires_grad_(True)
        if ddx is not None:
            ddx.requires_grad_(True)
        z = self.encoder(x)
        x_decode = self.decoder(z)



        encoder_weights = [layer.weight for layer in self.encoder if isinstance(layer, nn.Linear)]
        encoder_biases = [layer.bias for layer in self.encoder if isinstance(layer, nn.Linear)]

        # Extract weights and biases from decoder
        decoder_weights = [layer.weight for layer in self.decoder if isinstance(layer, nn.Linear)]
        decoder_biases = [layer.bias for layer in self.decoder if isinstance(layer, nn.Linear)]

        # print("Encoder Weights Shapes:")
        # for weight in encoder_weights:
        #     print(weight.shape)

        # print("Decoder Weights Shapes:")
        # for weight in decoder_weights:
        #     print(weight.shape)

        if self.model_order == 1:
        # Initialize a Jacobian tensor of the correct shape
            dz = z_derivative(x, dx, encoder_weights, encoder_biases, self.activation)
            # print("z",z)
            Theta = self.sindy_library(z)

            if self.sequential_thresholding:
                sindy_predict = torch.matmul(Theta, self.coefficient_mask * self.sindy_coefficients)
            else:
                sindy_predict = torch.matmul(Theta, self.sindy_coefficients)
            dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, self.activation)
            return z, x_decode, dz, dx_decode, sindy_predict,x, dx, self.sindy_coefficients
        
        elif self.model_order == 2:
            dz = z_derivative(x, dx, encoder_weights, encoder_biases, self.activation)
            # dz = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), create_graph=True, retain_graph=True)[0]
            ddz = z_derivative_order2(x, dx, ddx, encoder_weights, encoder_biases, self.activation)
            # ddz = torch.autograd.grad(outputs=dz, inputs=x, grad_outputs=torch.ones_like(dz), create_graph=True, retain_graph=True)[0]
            Theta = self.sindy_library_order2(z, dz)
            print("theta", Theta.shape)
            print(self.sindy_coefficients.shape, self.sindy_coefficients.shape)
            if self.sequential_thresholding:
                sindy_predict = torch.matmul(Theta, self.coefficient_mask * self.sindy_coefficients)
            else:
                sindy_predict = torch.matmul(Theta, self.sindy_coefficients)
       #     dx_decode, ddx_decode = torch.autograd.grad(z, dx, grad_outputs=sindy_predict, create_graph=True)[0], torch.autograd.grad(dz, dx, grad_outputs=sindy_predict, create_graph=True)[0]
            # dx_decode = torch.autograd.grad(outputs=sindy_predict, inputs=z, grad_outputs=torch.ones_like(sindy_predict), create_graph=True, retain_graph=True)[0]
            # ddx_decode = torch.autograd.grad(outputs=sindy_predict, inputs=dz, grad_outputs=torch.ones_like(sindy_predict), create_graph=True, retain_graph=True)[0]
            print("z",z.shape)
            dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, self.activation)
            ddx_decode = z_derivative_order2(z, sindy_predict, ddz, decoder_weights, decoder_biases, self.activation)
        
            return z, x_decode, dz, dx_decode, ddz, x, dx,self.sindy_coefficients, ddx_decode, sindy_predict, ddx
        else:
            raise ValueError('Invalid model order')
        
    def define_loss(self, x, dx, ddx=None, params=None):
        if params['model_order'] == 1:
            z, x_decode, dz, dx_decode, sindy_predict, x, dx,sindy_coefficients = self.forward(x,dx)[:8]
        if params['model_order'] == 2:
            z, x_decode, dz, dx_decode, ddz, x, dx,sindy_coefficients, ddx_decode, sindy_predict, ddx = self.forward(x,dx,ddx)[:11]
            # ddz, ddx_decode, ddx = network_output[8:11]
        
        # print(f"z shape: {z.shape}")
        # print(f"x_decode shape: {x_decode.shape}")
        # print(f"dz shape: {dz.shape}")
        # print(f"dx_decode shape: {dx_decode.shape}")
        # print(f"sindy_predict shape: {sindy_predict.shape}")

        losses = {}
        losses['decoder'] = F.mse_loss(x_decode, x)
        losses['sindy_z'] = F.mse_loss(sindy_predict,dz )
        # print("losses[sindy_z]:",losses['sindy_z'])
        losses['sindy_x'] = F.mse_loss(dx_decode, dx)
        # print("dx_decode none zero:",torch.nonzero(dx_decode))
        # print("losses[sindy_x]:",losses['sindy_x'])

        if params['model_order'] == 2:
            losses['sindy_z'] = F.mse_loss(ddz, sindy_predict)
            losses['sindy_x'] = F.mse_loss(ddx_decode, ddx)

        self.coefficient_mask = torch.nn.Parameter(params['coefficient_mask'], requires_grad=False)
        # sindy_coefficients = network_output[4] * params['coefficient_mask']
        # Move sindy_coefficients to CPU for numpy operations
        sindy_coefficients_cpu = sindy_coefficients.cpu()
        #print(f"sindy_coefficients_cpu shape: {sindy_coefficients_cpu.shape}")
            # Ensure params['coefficient_mask'] is a tensor
        if isinstance(params['coefficient_mask'], np.ndarray):
            params_cpu = torch.tensor(params['coefficient_mask'], device='cpu')
        else:
            params_cpu = params['coefficient_mask'].cpu()
        #print(f"params_cpu shape: {params_cpu.shape}")
        # Assuming params['coefficient_mask'] is a tensor
        sindy_coefficients = sindy_coefficients_cpu * params_cpu

        losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))

        loss = params['loss_weight_decoder'] * losses['decoder'] \
            + params['loss_weight_sindy_z'] * losses['sindy_z'] \
            + params['loss_weight_sindy_x'] * losses['sindy_x'] \
            + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

        loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                        + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                        + params['loss_weight_sindy_x'] * losses['sindy_x']

        return loss, losses, loss_refinement
        
        
def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D torch tensor, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of torch tensors containing the network weights
        biases - List of torch tensors containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Torch tensor, first order time derivatives of the network output.
    """
    dz = dx.clone()  # Clone dx to start the derivative computation
    activation_fn = {
        'elu': torch.nn.functional.elu,
        'relu': torch.nn.functional.relu,
        'sigmoid': torch.sigmoid,
        'linear': lambda x: x
    }[activation]

    for i in range(len(weights)-1):
        input = torch.matmul(input, weights[i].transpose(0,1)) + biases[i]
        if activation == 'elu':
            dz = torch.multiply(torch.minimum(torch.exp(input), torch.tensor(1.0, device=input.device)), torch.matmul(dz, weights[i]))
            input = activation_fn(input)
        elif activation == 'relu':
            dz = torch.multiply((input > 0).float(), torch.matmul(dz, weights[i].transpose(0,1)))
            input = activation_fn(input)
        elif activation == 'sigmoid':
            input = activation_fn(input)
            dz = torch.multiply(input * (1 - input), torch.matmul(dz, weights[i].transpose(0,1)))
        

    dz = torch.matmul(dz, weights[-1].transpose(0,1))
    return dz

def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    """
    Compute the second order time derivatives by propagating through the network.

    Arguments:
        input - 2D torch tensor, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of torch tensors containing the network weights
        biases - List of torch tensors containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        ddz - Torch tensor, second order time derivatives of the network output.
    """
    dz = dx.clone()  # Clone dx to start the first derivative computation
    ddz = ddx.clone()  # Clone ddx to start the second derivative computation
    activation_fn = {
        'elu': torch.nn.functional.elu,
        'relu': torch.nn.functional.relu,
        'sigmoid': torch.sigmoid,
        'linear': lambda x: x
    }[activation]

    for i in range(len(weights)-1):
        input = torch.matmul(input, weights[i].transpose(0,1)) + biases[i]
        print('ddz',ddz.shape, weights[i].shape)
        if activation == 'elu':
            exp_input = torch.exp(input)
            elu_derivative = torch.minimum(exp_input, torch.tensor(1.0, device=input.device))
            dz = torch.multiply(elu_derivative, torch.matmul(dz, weights[i].transpose(0,1)))
            ddz = torch.multiply(elu_derivative, torch.matmul(ddz, weights[i].transpose(0,1))) + torch.multiply(exp_input * dz, torch.matmul(dz, weights[i].transpose(0,1)))
            input = activation_fn(input)
        elif activation == 'relu':
            relu_derivative = (input > 0).float()
            dz = torch.multiply(relu_derivative, torch.matmul(dz, weights[i].transpose(0,1)))
            ddz = torch.multiply(relu_derivative, torch.matmul(ddz, weights[i].transpose(0,1))) + torch.multiply(relu_derivative, torch.matmul(dz, weights[i].transpose(0,1)))
            input = activation_fn(input)
        elif activation == 'sigmoid':
            input = activation_fn(input)
            sigmoid_derivative = input * (1 - input)
            # dz = torch.multiply(sigmoid_derivative, torch.matmul(dz, weights[i].transpose(0,1)))
            # ddz = torch.multiply(sigmoid_derivative, torch.matmul(ddz, weights[i].transpose(0,1))) + torch.multiply(sigmoid_derivative * (1 - 2 * input), torch.matmul(dz, weights[i].transpose(0,1)))
            dz_temp = torch.matmul(dz, weights[i].transpose(0, 1))
            ddz_temp = torch.matmul(ddz, weights[i].transpose(0, 1))
            term1 = torch.multiply(sigmoid_derivative, ddz_temp)
            term2 = torch.multiply(sigmoid_derivative * (1 - 2 * input), dz_temp)
            print(f"sigmoid_derivative shape: {sigmoid_derivative.shape}")
            print(f"ddz_temp shape: {ddz_temp.shape}")
            print(f"dz_temp shape: {dz_temp.shape}")
            print(f"term1 shape: {term1.shape}")
            print(f"term2 shape: {term2.shape}")
            dz = torch.multiply(sigmoid_derivative, torch.matmul(dz, weights[i].transpose(0,1)))
            ddz = term1 + term2

    ddz = torch.matmul(ddz, weights[-1].transpose(0,1))
    return ddz

def compute_jacobian_on_cpu(x_decode, z):
    batch_size, x_dim, z_dim = x_decode.shape[0], x_decode.shape[1], z.shape[1]
    dx_decode_dz = torch.zeros(batch_size, x_dim, z_dim, device='cpu')  # Allocate on CPU

    # Compute the Jacobian manually for each output in x_decode
    for i in range(x_dim):  # Loop over each dimension of x_decode
        grad_outputs = torch.zeros_like(x_decode)
        grad_outputs[:, i] = 1  # Set one component of the output to 1 at a time
        # Move grad_outputs to CPU
        grad_outputs_cpu = grad_outputs.to('cpu')
        x_decode_cpu = x_decode.to('cpu')
        z_cpu = z.to('cpu')

        dx_decode_dz_i = torch.autograd.grad(
            outputs=x_decode_cpu, inputs=z_cpu, grad_outputs=grad_outputs_cpu,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]

        if dx_decode_dz_i is None:
            print(f"No gradient computed for output dimension {i}")
        else:
            print(f"Gradient for output dimension {i}:\n{dx_decode_dz_i}")

        if dx_decode_dz_i is not None:
            dx_decode_dz[:, i, :] = dx_decode_dz_i

    # Ensure the gradients are back on GPU after computation
    dx_decode_dz = dx_decode_dz.to(x_decode.device)
    return dx_decode_dz 




# def define_loss(network_output, params):
#     if params['model_order'] == 1:
#         z, x_decode, dz, dx_decode, sindy_predict, x, dx,sindy_coefficients = network_output[:8]
#     if params['model_order'] == 2:
#         z, x_decode, dz, dx_decode, ddz, x, dx,sindy_coefficients, ddx_decode, sindy_predict, ddx = network_output[:11]
#         ddz, ddx_decode, ddx = network_output[8:11]
    
#     # print(f"z shape: {z.shape}")
#     # print(f"x_decode shape: {x_decode.shape}")
#     # print(f"dz shape: {dz.shape}")
#     # print(f"dx_decode shape: {dx_decode.shape}")
#     # print(f"sindy_predict shape: {sindy_predict.shape}")

#     losses = {}
#     losses['decoder'] = F.mse_loss(x_decode, x)
#     losses['sindy_z'] = F.mse_loss(sindy_predict,dz )
#     # print("losses[sindy_z]:",losses['sindy_z'])
#     losses['sindy_x'] = F.mse_loss(dx_decode, dx)
#     # print("dx_decode none zero:",torch.nonzero(dx_decode))
#     # print("losses[sindy_x]:",losses['sindy_x'])

#     if params['model_order'] == 2:
#         losses['sindy_z'] = F.mse_loss(ddz, sindy_predict)
#         losses['sindy_x'] = F.mse_loss(ddx_decode, ddx)

#     # sindy_coefficients = network_output[4] * params['coefficient_mask']
#     # Move sindy_coefficients to CPU for numpy operations
#     sindy_coefficients_cpu = sindy_coefficients.cpu()
#     #print(f"sindy_coefficients_cpu shape: {sindy_coefficients_cpu.shape}")
#         # Ensure params['coefficient_mask'] is a tensor
#     if isinstance(params['coefficient_mask'], np.ndarray):
#         params_cpu = torch.tensor(params['coefficient_mask'], device='cpu')
#     else:
#         params_cpu = params['coefficient_mask'].cpu()
#     #print(f"params_cpu shape: {params_cpu.shape}")
#     # Assuming params['coefficient_mask'] is a tensor
#     sindy_coefficients = sindy_coefficients_cpu * params_cpu

#     losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))

#     loss = params['loss_weight_decoder'] * losses['decoder'] \
#            + params['loss_weight_sindy_z'] * losses['sindy_z'] \
#            + params['loss_weight_sindy_x'] * losses['sindy_x'] \
#            + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

#     loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
#                       + params['loss_weight_sindy_z'] * losses['sindy_z'] \
#                       + params['loss_weight_sindy_x'] * losses['sindy_x']

#     return loss, losses, loss_refinement

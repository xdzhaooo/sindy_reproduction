import numpy as np
import torch 
import torch.optim as optim
import pickle
from autoencoder import FullNetwork

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['coefficient_mask'] = params['coefficient_mask']
    model = FullNetwork(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    print("val_data", val_data["ddx"])
    # Prepare validation data
    validation_data = prepare_data(val_data, params, device)

    x_tensor = torch.tensor(val_data['x'], dtype=torch.float32).to(device)
    x_norm = torch.mean(x_tensor**2)
    # Assuming val_data['dx'] and val_data['ddx'] are NumPy arrays
    val_data['dx'] = torch.tensor(val_data['dx'], dtype=torch.float32).to(device)
    if 'ddx' in val_data and val_data['ddx'] is not None:
        val_data['ddx'] = torch.tensor(val_data['ddx'], dtype=torch.float32).to(device)

    # Compute sindy_predict_norm_x on the GPU
    sindy_predict_norm_x = torch.mean(val_data['dx']**2).item() if params['model_order'] == 1 else torch.mean(val_data['ddx']**2).item()

    #sindy_predict_norm_x = np.mean(val_data['dx']**2).item() if params['model_order'] == 1 else np.mean(val_data['ddx']**2).item()

    validation_losses = []
    sindy_model_terms = [torch.sum(torch.tensor(params['coefficient_mask'])).cpu().numpy()]


    print('TRAINING')
    model.train()
    for epoch in range(params['max_epochs']):
        # model.train()
        # print('Epoch %d' % epoch)
        for j in range(params['epoch_size'] // params['batch_size']):
            model.train()
            #print("params['epoch_size'] // params['batch_size']" , params['epoch_size'] // params['batch_size'])
            #print('Batch %d' % j)
            # print("j", j)
            batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
            #print("batch_idxs", batch_idxs)
            train_dict = prepare_data(training_data, params, device, idxs=batch_idxs)
            # print("train_dict", train_dict['x'][0])
            x_batch, dx_batch, ddx_batch = train_dict['x'], train_dict['dx'], train_dict.get('ddx')
            # coefficient_mask = train_dict.get('coefficient_mask')

            # x_batch, dx_batch, ddx_batch = train_dict['x'], train_dict['dx'], train_dict.get('ddx')
            
            optimizer.zero_grad()
            # outputs = model(x_batch, dx_batch, ddx_batch)
           #  loss, losses, _ = define_loss(outputs, params)
            loss, losses, _ = model.define_loss(x_batch, dx_batch, ddx_batch, params)
            loss.backward()
            optimizer.step()

        if params['print_progress'] and (epoch % params['print_frequency'] == 0):
            # print("params['print_progress'] and (epoch % params['print_frequency'] == 0)")
            # model.eval()
            with torch.no_grad():
                validation_losses.append(print_progress(model, epoch, params, x_norm, sindy_predict_norm_x, val_data,training_data))

        if params['sequential_thresholding'] and (epoch % params['threshold_frequency'] == 0) and (epoch > 0):
            # print("params['sequential_thresholding'] and (epoch % params['threshold_frequency'] == 0) and (epoch > 0)")
            # model.eval()
            #with torch.no_grad():
            # torch.abs(model.sindy_coefficients) > params['coefficient_threshold']
            params['coefficient_mask'] = torch.abs(model.sindy_coefficients) > params['coefficient_threshold']
            validation_data['coefficient_mask'] = params['coefficient_mask']
            sindy_model_terms.append(torch.sum(params['coefficient_mask']).cpu().numpy())
            # print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))

    print('REFINEMENT')
    for epoch in range(params['refinement_epochs']):
        # model.train()
        for j in range(params['epoch_size'] // params['batch_size']):
            batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
            train_dict = prepare_data(training_data, params,device, idxs=batch_idxs)
            x_batch, dx_batch, ddx_batch = train_dict['x'], train_dict['dx'], train_dict.get('ddx')
            # coefficient_mask = train_dict.get('coefficient_mask')

            optimizer.zero_grad()
            loss, losses, _ = model.define_loss(x_batch, dx_batch, ddx_batch, params)
            loss.backward()
            optimizer.step()

        if params['print_progress'] and (epoch % params['print_frequency'] == 0):

            with torch.no_grad():
                validation_losses.append(print_progress(model, epoch, params, x_norm, sindy_predict_norm_x, validation_data,train_dict))

    # print(params['coefficient_mask'])
    # for key in model.state_dict():
    #     print(key)
    #     print(model.state_dict()['coefficient_mask'])

    torch.save(model.state_dict(), params['data_path'] + params['save_name']+  '.pth')
    with open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    # model.eval()
    with torch.no_grad():
        model.eval()
       #  loss_val, final_losses, _ = define_loss(model(val_data['x'], val_data['dx'], val_data.get('ddx')), params)
        loss_val, final_losses, _ = model.define_loss(val_data['x'], val_data['dx'], val_data.get('ddx'), params)
        # final_losses = {name: loss.item() for name, loss in losses.items()}
        if params['model_order'] == 1:
            sindy_predict_norm_z = torch.mean(model(val_data['x'], val_data['dx'])[7]**2).item()
        else:
            sindy_predict_norm_z = torch.mean(model(val_data['x'], val_data['dx'], val_data['ddx'])[7]**2).item()
        # sindy_coefficients = model.sindy_coefficients.cpu().numpy()
        sindy_coefficients = model.sindy_coefficients.detach().cpu().numpy()

        results_dict = {
            'num_epochs': epoch,
            'x_norm': x_norm,
            'sindy_predict_norm_x': sindy_predict_norm_x,
            'sindy_predict_norm_z': sindy_predict_norm_z,
            'sindy_coefficients': sindy_coefficients,
            'loss_decoder': final_losses['decoder'],
            'loss_decoder_sindy': final_losses['sindy_x'],
            'loss_sindy': final_losses['sindy_z'],
            'loss_sindy_regularization': final_losses['sindy_regularization'],
            'validation_losses': np.array(validation_losses),
            'sindy_model_terms': np.array(sindy_model_terms)
        }

    return results_dict

def print_progress(model, epoch, params, x_norm, sindy_predict_norm, validation_dict, train_dict):
    """
    Print loss function values to keep track of the training progress.

    Arguments:
        model - the PyTorch model
        epoch - the training iteration
        params - dictionary of training parameters
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.
        validation_data - dictionary of validation data tensors

    Returns:
        Tuple of losses calculated on the validation set.
    """
    # model.eval()
    # with torch.no_grad():
    #     x_val, dx_val, ddx_val = validation_data['x'], validation_data['dx'], validation_data.get('ddx')

    #     # Ensure inputs require gradients
    #     x_val.requires_grad = True
    #     if dx_val is not None:
    #         dx_val.requires_grad = True
    #     if ddx_val is not None:
    #         ddx_val.requires_grad = True

    #     outputs = model(x_val, dx_val, ddx_val)
    #     loss, losses, _ = define_loss(outputs, params)
    #     loss_vals = {name: loss.item() for name, loss in losses.items()}
    #     print(f"Epoch {epoch}")
    #     print(f"   validation loss {loss_vals}")
    #     decoder_losses = (losses['decoder'].item(), losses['sindy_x'].item())
    #     loss_ratios = (decoder_losses[0] / x_norm, decoder_losses[1] / sindy_predict_norm)
    #     print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
    #     return loss_vals
    model.eval()
    with torch.no_grad():
       #  validation_data = model(validation_dict['x'], validation_dict['dx'], validation_dict.get('ddx'))
        val_total_loss, val_losses, _ = model.define_loss(
            validation_dict['x'], validation_dict['dx'], validation_dict.get('ddx'),
            params=params
       )
       #  train_data = model(train_dict['x'], train_dict['dx'], train_dict.get('ddx'))
        train_total_loss, train_losses, _ = model.define_loss(
            train_dict['x'], train_dict['dx'], train_dict.get('ddx'),
            params=params
        )
        # Output the losses for training data
        # print(f"Epoch {epoch}")
        # print(f"   Training Total Loss: {train_total_loss.item()}")
        # for loss_name, loss_val in train_losses.items():
        #     print(f"   Training {loss_name} Loss: {loss_val.item()}")

        # # Output the losses for validation data
        # print(f"   Validation Total Loss: {val_total_loss.item()}")
        # for loss_name, loss_val in val_losses.items():
        #     print(f"   Validation {loss_name} Loss: {loss_val.item()}")
        print(f"Epoch {epoch}")
    
        # 打印训练损失
        print("   Training Losses:")
        print(f"      Total Loss: {train_total_loss.item()}")
        for loss_name, loss_val in train_losses.items():
            print(f"      {loss_name} Loss: {loss_val.item()}")

        # 打印验证损失
        print("   Validation Losses:")
        print(f"      Total Loss: {val_total_loss.item()}")
        for loss_name, loss_val in val_losses.items():
            print(f"      {loss_name} Loss: {loss_val.item()}")
        # Calculate and display loss ratios
        decoder_loss_ratio = val_losses['decoder'].item() / x_norm
        sindy_x_loss_ratio = val_losses['sindy_x'].item() / sindy_predict_norm
        print(f"Decoder Loss Ratio: {decoder_loss_ratio:.6f}, Decoder SINDy Loss Ratio: {sindy_x_loss_ratio:.6f}")

        model.train()
        return {name: loss.item() for name, loss in val_losses.items()}

def prepare_data(data, params, device, idxs=None):
    """
    Prepare data for passing into PyTorch model.

    Arguments:
        data - dictionary containing the data to be passed in.
        params - dictionary containing model and training parameters.
        idxs - optional array of indices to select examples from the dataset.

    Returns:
        Dictionary containing the relevant data tensors.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    data_dict = {
        'x': torch.tensor(data['x'][idxs], dtype=torch.float32).to(device),
        'dx': torch.tensor(data['dx'][idxs], dtype=torch.float32).to(device)
    }
    # print("data_dict['x']", data_dict['x'])
    # print("data['x']", data['x'])
    if params['model_order'] == 2:
        data_dict['ddx'] = torch.tensor(data['ddx'][idxs], dtype=torch.float32).to(device)
    if params['sequential_thresholding']:
        data_dict['coefficient_mask'] = torch.tensor(params['coefficient_mask'], dtype=torch.float32).to(device)
    data_dict['learning_rate'] = torch.tensor(params['learning_rate'], dtype=torch.float32).to(device)
    return data_dict



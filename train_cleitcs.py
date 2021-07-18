import os
import torch
from collections import defaultdict
from ae import AE
from evaluation_utils import model_save_check, evaluate_target_regression_epoch
from loss_and_metrics import masked_simse,masked_mse,contrastive_loss
from multi_out_mlp import MoMLP
from encoder_decoder import EncoderDecoder
from copy import deepcopy
from mlp import MLP
from itertools import chain


def cleit_train_step(model, reference_encoder, transmitter, s_batch, t_batch, device, optimizer, alpha, history, scheduler=None):
    model.zero_grad()
    model.train()
    transmitter.zero_grad()
    reference_encoder.zero_grad()
    transmitter.train()
    reference_encoder.eval()

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device)

    t_x = t_batch[0].to(device)
    t_y = t_batch[1].to(device)

    x_m_code = transmitter(model.encoder(t_x))
    x_g_code = reference_encoder(s_x)

    code_loss = contrastive_loss(y_true=x_g_code, y_pred=x_m_code, device=device)
    loss = masked_simse(preds=model(t_x), labels=t_y) + alpha * code_loss

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['loss'].append(loss.cpu().detach().item())
    history['code_loss'].append(code_loss.cpu().detach().item())

    return history


def train_cleitcs(s_dataloaders, t_dataloaders, val_dataloader, test_dataloader, metric_name, seed, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders
    t_train_dataloader = t_dataloaders

    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    # get reference encoder
    aux_ae = deepcopy(autoencoder)

    aux_ae.encoder.load_state_dict(torch.load(os.path.join('./model_save/ae', f'ft_encoder_{seed}.pt')))
    print('reference encoder loaded')
    reference_encoder = aux_ae.encoder

    # construct transmitter
    transmitter = MLP(input_dim=kwargs['latent_dim'],
                      output_dim=kwargs['latent_dim'],
                      hidden_dims=[kwargs['latent_dim']]).to(kwargs['device'])

    encoder = autoencoder.encoder
    target_decoder = MoMLP(input_dim=kwargs['latent_dim'],
                           output_dim=kwargs['output_dim'],
                           hidden_dims=kwargs['regressor_hidden_dims'],
                           out_fn=torch.nn.Sigmoid).to(kwargs['device'])

    target_regressor = EncoderDecoder(encoder=encoder,
                                      decoder=target_decoder).to(kwargs['device'])

    train_history = defaultdict(list)
    # ae_eval_train_history = defaultdict(list)
    val_history = defaultdict(list)
    s_target_regression_eval_train_history = defaultdict(list)
    t_target_regression_eval_train_history = defaultdict(list)
    target_regression_eval_val_history = defaultdict(list)
    target_regression_eval_test_history = defaultdict(list)
    cleit_params = [
        target_regressor.parameters(),
        transmitter.parameters()
    ]
    model_optimizer = torch.optim.AdamW(chain(*cleit_params), lr=kwargs['lr'])
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'Coral training epoch {epoch}')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            train_history = cleit_train_step(model=target_regressor,
                                             transmitter=transmitter,
                                             reference_encoder=reference_encoder,
                                             s_batch=s_batch,
                                             t_batch=t_batch,
                                             device=kwargs['device'],
                                             optimizer=model_optimizer,
                                             alpha=kwargs['alpha'],
                                             history=train_history)
        s_target_regression_eval_train_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                                  dataloader=s_train_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=s_target_regression_eval_train_history)

        t_target_regression_eval_train_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                                  dataloader=t_train_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=t_target_regression_eval_train_history)
        target_regression_eval_val_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                              dataloader=val_dataloader,
                                                                              device=kwargs['device'],
                                                                              history=target_regression_eval_val_history)
        target_regression_eval_test_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                               dataloader=test_dataloader,
                                                                               device=kwargs['device'],
                                                                               history=target_regression_eval_test_history)

        save_flag, stop_flag = model_save_check(history=target_regression_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=50)
        if save_flag:
            torch.save(target_regressor.state_dict(), os.path.join(kwargs['model_save_folder'], f'cleitcs_regressor_{seed}.pt'))
        if stop_flag:
            break
    target_regressor.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], f'cleitcs_regressor_{seed}.pt')))

    # evaluate_target_regression_epoch(regressor=target_regressor,
    #                                  dataloader=val_dataloader,
    #                                  device=kwargs['device'],
    #                                  history=None,
    #                                  seed=seed,
    #                                  output_folder=kwargs['model_save_folder'])
    evaluate_target_regression_epoch(regressor=target_regressor,
                                     dataloader=test_dataloader,
                                     device=kwargs['device'],
                                     history=None,
                                     seed=seed,
                                     output_folder=kwargs['model_save_folder'])

    return target_regressor, (
        train_history, s_target_regression_eval_train_history, t_target_regression_eval_train_history,
        target_regression_eval_val_history, target_regression_eval_test_history)

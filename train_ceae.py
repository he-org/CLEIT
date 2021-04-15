import torch
import os
from collections import defaultdict
from itertools import chain
from mlp import MLP
from loss_and_metrics import contrastive_loss
from encoder_decoder import EncoderDecoder
from ceae import CEAE


def ceae_train_step(ae, transmitter, batch, device, optimizer, history, scheduler=None):
    ae.zero_grad()
    transmitter.zero_grad()
    ae.train()
    transmitter.train()

    x_p = batch[0].to(device)
    x_t = batch[1].to(device)
    loss_dict = ae.loss_function(*ae(x_p))
    optimizer.zero_grad()

    x_t_code = x_t#
    x_p_code = transmitter(ae.encode(x_p))

    code_loss = contrastive_loss(y_true=x_t_code, y_pred=x_p_code, device=device)
    # loss = loss_dict['loss']
    # loss = loss_dict['loss'] + code_loss
    optimizer.zero_grad()

    code_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    for k, v in loss_dict.items():
        history[k].append(v)
    history['code_loss'].append(code_loss.cpu().detach().item())
    return history


def train_ceae(dataloader, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    autoencoder = CEAE(input_dim=kwargs['input_dim'],
                       latent_dim=50).to(kwargs['device'])

    # construct transmitter
    transmitter = MLP(input_dim=50,
                      output_dim=50,
                      hidden_dims=[50]).to(kwargs['device'])

    ae_eval_train_history = defaultdict(list)
    ae_eval_test_history = defaultdict(list)

    ceae_params = [
        autoencoder.parameters(),
        transmitter.parameters()
    ]
    ceae_optimizer = torch.optim.AdamW(chain(*ceae_params), lr=kwargs['lr'])
    # start autoencoder pretraining
    for epoch in range(int(kwargs['train_num_epochs'])):
        for step, batch in enumerate(dataloader):
            ae_eval_train_history = ceae_train_step(ae=autoencoder,
                                                    transmitter=transmitter,
                                                    batch=batch,
                                                    device=kwargs['device'],
                                                    optimizer=ceae_optimizer,
                                                    history=ae_eval_train_history)
        if epoch % 50 == 0:
            print(f'----CE Autoencoder Training Epoch {epoch} ----')
            torch.save(autoencoder.encoder.state_dict(),
                       os.path.join(kwargs['model_save_folder'], f'train_epoch_{epoch}_encoder.pt'))
            torch.save(transmitter.state_dict(),
                       os.path.join(kwargs['model_save_folder'], f'train_epoch_{epoch}_transmitter.pt'))
    encoder = EncoderDecoder(encoder=autoencoder.encoder,
                             decoder=transmitter).to(kwargs['device'])
    #
    # torch.save(encoder.state_dict(),
    #            os.path.join(kwargs['model_save_folder'], f'train_epoch_{epoch}_encoder.pt'))

    return encoder, (ae_eval_train_history, ae_eval_test_history)

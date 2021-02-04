import torch
import os
from collections import defaultdict
from vae import VAE
from mask_mlp import MaskMLP
from encoder_decoder import EncoderDecoder
from loss_and_metrics import contrastive_loss


def cleit_train_step(ae, reference_encoder, batch, device, optimizer, history, scheduler=None):
    ae.zero_grad()
    reference_encoder.zero_grad()
    ae.train()
    reference_encoder.eval()

    x_m = batch[0].to(device)
    x_g = batch[1].to(device)
    loss_dict = ae.loss_function(*ae(x_m))
    optimizer.zero_grad()

    x_m_code = ae.encoder(x_m)
    x_g_code = reference_encoder(x_g)

    code_loss = contrastive_loss(y_true=x_g_code, y_pred=x_m_code)
    loss = loss_dict['loss'] + code_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    for k, v in loss_dict.items():
        history[k].append(v)

    return history


def train_cleit(dataloader, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """

    autoencoder = VAE(input_dim=kwargs['input_dim'],
                      latent_dim=kwargs['latent_dim'],
                      hidden_dims=kwargs['encoder_hidden_dims'],
                      dop=kwargs['dop']).to(kwargs['device'])

    # get reference encoder
    target_decoder = MaskMLP(input_dim=kwargs['latent_dim'],
                             output_dim=kwargs['output_dim'],
                             hidden_dims=kwargs['regressor_hidden_dims']).to(kwargs['device'])
    target_regressor = EncoderDecoder(encoder=autoencoder.encoder,
                                      decoder=target_decoder).to(kwargs['device'])
    target_regressor.load_state_dict(torch.load(os.path.join('./model_save', 'target_regressor.pt')))

    reference_encoder = target_regressor.encoder

    ae_eval_train_history = defaultdict(list)
    ae_eval_test_history = defaultdict(list)

    if kwargs['retrain_flag']:
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
        # start autoencoder pretraining
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'----Autoencoder Training Epoch {epoch} ----')
            for step, batch in enumerate(dataloader):
                ae_eval_train_history = cleit_train_step(ae=autoencoder,
                                                         reference_encoder=reference_encoder,
                                                         batch=batch,
                                                         device=kwargs['device'],
                                                         optimizer=ae_optimizer,
                                                         history=ae_eval_train_history)
        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'cleit_vae.pt'))
    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cleit_vae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_eval_train_history, ae_eval_test_history)

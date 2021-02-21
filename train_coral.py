import os
import torch
from loss_and_metrics import cov
from collections import defaultdict
from ae import AE
from evaluation_utils import model_save_check
from loss_and_metrics import masked_simse


def coral_train_step(model, s_batch, t_batch, device, optimizer, alpha, history, scheduler=None):
    model.zero_grad()
    model.train()

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device)

    t_x = t_batch[0].to(device)
    t_y = t_batch[1].to(device)

    s_code = model.encode(s_x)
    t_code = model.encode(t_x)

    s_cov = cov(s_code)
    t_cov = cov(t_code)

    coral_loss = (torch.square(torch.norm(s_cov - t_cov, p='fro'))) / (4 * (s_code.size()[-1] ** 2))
    # coral_loss = torch.square(torch.norm(s_cov-t_cov, p='fro'))

    loss = masked_simse(preds=model(s_x), labels=s_y) + masked_simse(preds=model(t_x), labels=t_y) + alpha * coral_loss

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['loss'].append(loss.cpu().detach().item())
    history['coral_loss'].append(coral_loss.cpu().detach().item())

    return history


def train_coral(s_dataloaders, t_dataloaders, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders
    s_test_dataloader = s_dataloaders

    t_train_dataloader = t_dataloaders
    t_test_dataloader = t_dataloaders

    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])


    ae_train_history = defaultdict(list)
    # ae_eval_train_history = defaultdict(list)
    ae_eval_val_history = defaultdict(list)


    if kwargs['retrain_flag']:
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'AE training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                ae_train_history = coral_ae_train_step(ae=autoencoder,
                                                       s_batch=s_batch,
                                                       t_batch=t_batch,
                                                       device=kwargs['device'],
                                                       optimizer=ae_optimizer,
                                                       alpha=kwargs['alpha'],
                                                       history=ae_train_history)
            ae_eval_val_history = eval_ae_epoch(ae=autoencoder,
                                                s_dataloader=s_test_dataloader,
                                                t_dataloader=t_test_dataloader,
                                                device=kwargs['device'],
                                                history=ae_eval_val_history)
            save_flag, stop_flag = model_save_check(ae_eval_val_history, metric_name='loss', tolerance_count=50)
            if kwargs['es_flag']:
                if save_flag:
                    torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'coral_ae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'coral_ae.pt')))

        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'coral_ae.pt'))

    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'coral_ae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_train_history, ae_eval_val_history)

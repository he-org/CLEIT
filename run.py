from tensorflow import keras
import tensorflow as tf
import data
import data_config
import preprocess_ccle_gdsc_utils
import preprocess_xena_utils
import loss
import train
from module import AE, VAE
import os
import pickle
import utils

if __name__ == '__main__':
    data_provider = data.DataProvider(feature_filter_fn=preprocess_xena_utils.filter_with_MAD, feature_number=10000,
                                      omics=['gex', 'mut'], scale_fn=data.min_max_scale)

    gex_auto_encoder = VAE(latent_dim=128,
                           output_dim=data_provider.shape_dict['gex'],
                           architecture=[1024, 512, 256],
                           noise_fn=keras.layers.GaussianNoise,
                           output_act_fn=keras.activations.sigmoid)

    tf.keras.backend.clear_session()
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.unlabeled_data['gex'].values, data_provider.unlabeled_data['gex'].values))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['gex'].values, data_provider.labeled_data['gex'].values))

    gex_encoder, gex_pre_train_history_df = train.pre_train_gex_AE(auto_encoder=gex_auto_encoder, train_dataset=train_dataset,
                                                     val_dataset=val_dataset, max_epochs=100)

    gex_fine_tune_train_history, gex_fine_tune_validation_history = train.fine_tune_gex_encoder(
        gex_encoder,
        target_df=data_provider.labeled_data['target'],
        raw_X=data_provider.labeled_data['gex'])

    mut_auto_encoder = VAE(latent_dim=128,
                           output_dim=data_provider.shape_dict['mut'],
                           architecture=[1024, 512, 256],
                           noise_fn=keras.layers.GaussianNoise,
                           output_act_fn=keras.activations.sigmoid)

    tf.keras.backend.clear_session()
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].values,
         data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].values,
         data_provider.unlabeled_data['gex'].loc[data_provider.matched_index].values))

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['mut'].values,
         data_provider.labeled_data['mut'].values,
         data_provider.labeled_data['gex'].values))

    mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset,transmission_loss_fn=loss.contrastive_loss)
    #mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset,transmission_loss_fn=loss.mmd_loss)
    #mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE_with_GAN(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset)


    mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
        mut_encoder,reference_encoder=gex_encoder,
        target_df=data_provider.labeled_data['target'],
        raw_X=data_provider.labeled_data['mut'],raw_reference_X=data_provider.labeled_data['gex'], transmission_loss_fn=loss.contrastive_loss
    )

    #mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
    #    mut_encoder,reference_encoder=gex_encoder,
    #    target_df=data_provider.labeled_data['target'],
    #    raw_X=data_provider.labeled_data['mut'],raw_reference_X=data_provider.labeled_data['gex'], transmission_loss_fn=loss.mmd_loss)

    #mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder_with_GAN(
    #    mut_encoder,
    #    target_df=data_provider.labeled_data['target'],
    #    raw_X=data_provider.labeled_data['mut'])

    utils.safe_make_dir('history')

    with open(os.path.join('history', 'history.pkl'), 'wb') as handle:
        pickle.dump(gex_pre_train_history_df)
        pickle.dump(gex_fine_tune_train_history)
        pickle.dump(gex_fine_tune_validation_history)
        pickle.dump(mut_pre_train_history_df)
        pickle.dump(mut_fine_tune_train_history)
        pickle.dump(mut_fine_tune_validation_history)















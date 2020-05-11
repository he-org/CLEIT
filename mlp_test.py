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
import model_config
import argparse
import predict
import pandas as pd

from tensorflow import keras
from functools import partial


# gex encoder pre-training can be shared
# gex fine-tuning needs to be retrained for different training folds, one time for the mutation-only
# mutation pre-training needs to be retrained for different training folds * different transmission function, one time for each transmission function for the mutation only
# mutation fine-tuning needs to be retrained per pre-training, in addition to fixed encoders (no fine-tuning on encoders)
# no pre-training on encoder, simply fine tuning (labeled data).
# no transmission

def train_regressor(encoder,
                    regressor,
                    train_dataset,
                    test_data,
                    batch_size=model_config.batch_size,
                    loss_fn=loss.penalized_mean_squared_error,
                    max_epoch=model_config.max_epoch,
                    min_epoch=model_config.min_epoch,
                    gradient_threshold=model_config.gradient_threshold,
                    ):
    train_dataset = train_dataset.shuffle(buffer_size=512).batch(batch_size)
    lr = model_config.fine_tuning_lr
    for epoch in range(max_epoch):
        print(epoch)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        grad_norm = 0.
        counts = 0.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                encoded_X = encoder(x_batch_train, training=True)
                preds = regressor(encoded_X, training=True)
                loss_value = loss_fn(y_pred=preds, y_true=y_batch_train)
                loss_value += sum(encoder.losses)
                loss_value += sum(regressor.losses)
                grads = tape.gradient(loss_value, encoder.trainable_variables + regressor.trainable_variables)
                optimizer.apply_gradients(zip(grads, to_train_variables))
                grad_norm += tf.linalg.global_norm(grads)
                counts += 1
        if gradient_threshold is not None and grad_norm/counts < gradient_threshold:
            break
    encoded_X = encoder(test_data, training=False)
    predictions = regressor(encoded_X, training=False)

    return predictions.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLP regressor')
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--propagation', dest='propagation', action='store_true')
    parser.add_argument('--no-propagation', dest='propagation', action='store_false')
    parser.set_defaults(propagation=True)

    parser.add_argument('--target', dest='target', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--filter', dest='filter', nargs='?', default='FILE', choices=['MAD', 'FILE'])
    parser.add_argument('--feat_num', dest='feature_number', nargs='?', default=5000)

    parser.add_argument('--gpu', dest='gpu', type=int, nargs='?', default=0)
    parser.add_argument('--exp_type', dest='exp_type', nargs='?', default='cv', choices=['cv', 'test'])
    parser.add_argument('--dat_type', dest='dat_type', nargs='?', default='mut', choices=['gex', 'mut'])

    args = parser.parse_args()
    data_provider = data.DataProvider(feature_filter=args.filter, target=args.target,
                                      feature_number=args.feature_number,
                                      omics=['gex', 'mut'])

    # unlabeled gex has few small negative value
    data_provider.unlabeled_data['gex'].where(data_provider.unlabeled_data['gex'] > 0, 0, inplace=True)
    assert data_provider.unlabeled_data['gex'].min().min() >= 0

    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #    tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if args.exp_type == 'cv':

        prediction_df = pd.DataFrame(
            np.full_like(data_provider.labeled_data['target'], fill_value=-1),
            index=data_provider.labeled_data['target'].index,
            columns=data_provider.labeled_data['target'].columns)

        for i in range(len(data_provider.get_k_folds())):
            # gex fine tuning
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.labeled_data[args.dat_type].iloc[data_provider.get_k_folds()[i][0]].values,
                 data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][0]].values))

            test_data = data_provider.labeled_data[args.dat_type].iloc[data_provider.get_k_folds()[i][1]].values

            encoder = module.EncoderBlock(latent_dim=model_config.encoder_latent_dimension,
                                          architecture=model_config.encoder_architecture,
                                          output_act_fn=model_config.encoder_output_act_fn,
                                          act_fn=model_config.encoder_act_fn,
                                          kernel_regularizer_l=model_config.kernel_regularizer_l)

            regressor = module.MLPBlockWithMask(architecture=model_config.regressor_architecture,
                                                shared_layer_num=model_config.regressor_shared_layer_num,
                                                act_fn=model_config.regressor_act_fn,
                                                output_act_fn=model_config.regressor_output_act_fn,
                                                output_dim=model_config.regressor_output_dim)

            prediction_df.loc[data_provider.labeled_data[args.dat_type].iloc[data_provider.get_k_folds()[i][1]].index,
            :] = train_regressor(
                encoder=encoder,
                regressor=regressor,
                train_dataset=train_dataset,
                test_data=test_data
            )

            prediction_df.to_csv(
                os.path.join('predictions', args.target + '_' + args.dat_type + '_mlp_prediction.csv'),
                index_label='Sample')


    else:
        # gex fine tuning
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['mut'].values,
             data_provider.labeled_data['target'].values))

        test_data = data_provider.labeled_test_data['mut'].values

        encoder = module.EncoderBlock(latent_dim=model_config.encoder_latent_dimension,
                                      architecture=model_config.encoder_architecture,
                                      output_act_fn=model_config.encoder_output_act_fn,
                                      act_fn=model_config.encoder_act_fn,
                                      kernel_regularizer_l=model_config.kernel_regularizer_l)

        regressor = module.MLPBlockWithMask(architecture=model_config.regressor_architecture,
                                            shared_layer_num=model_config.regressor_shared_layer_num,
                                            act_fn=model_config.regressor_act_fn,
                                            output_act_fn=model_config.regressor_output_act_fn,
                                            output_dim=model_config.regressor_output_dim)

        mut_test_prediction_df = pd.DataFrame(np.full_like(data_provider.labeled_test_data['target'], fill_value=-1),
                                              index=data_provider.labeled_test_data['target'].index,
                                              columns=data_provider.labeled_test_data['target'].columns)

        mut_test_prediction_df.loc[data_provider.labeled_test_data['mut'].index, :] = train_regressor(
                encoder=encoder,
                regressor=regressor,
                train_dataset=train_dataset,
                test_data=test_data
        )

        mut_test_prediction_df.to_csv(os.path.join('predictions', args.target + '_mut_only_mlp_prediction.csv'),
                                      index_label='Sample')


import os
import argparse
import sys
from tensorflow import keras
from DL_ChannelCoding.utils import find_last_checkpoint, train_callbacks
from DL_ChannelCoding.utils import decoding_L2_loss as L2_loss
from DL_ChannelCoding.utils import decoding_L1_loss as L1_loss
from DL_ChannelCoding.utils import encoding_deviation_loss as dev_loss
from DL_ChannelCoding.utils import bit_error_rate as BER
from DL_ChannelCoding.utils import median_encoding_deviation as med_dev
from DL_ChannelCoding.utils import max_encoding_deviation as max_dev
from DL_ChannelCoding.model import get_training_model
from DL_ChannelCoding.generators import InfoGenerator

def parse_args(args):
    parser = argparse.ArgumentParser(description='Training script for communication system.')
    parser.add_argument('-checkpoint_path', type=str,
                        help='Path to load and store checkpoints of model while training')
    parser.add_argument('-resume_previous', action='store_true',
                        help='Resume training from latest previous model checkpoint')
    parser.add_argument('-gpu', type=int,
                        help='ID of gpu on the machine. Run nvidia-smi to see options')
    parser.add_argument('-epoch_num', type=int, default=100,
                        help='Number of epochs to preform')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('-info_len', type=int, default=15,
                        help='Size of information word at encoder input')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Size of batch at encoder input')
    parser.add_argument('-codeword_len', type=int, default=18,
                        help='Size of codeword')
    parser.add_argument('-noise_std', type=float, default=0.1,
                        help='Standard deviation of gaussian noise')
    parser.add_argument('-dense_nums', nargs=2, default=(3,3),
                        help='Numbers of dense layers in (encoder, decoder)')
    parser.add_argument('-num-generator_procs', type=int, default=1,
                        help='Number of generator processes.')

    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    if args.resume_previous:
        print("Resume latest flagged. Looking for checkpoints:")
        last_check = find_last_checkpoint(args.checkpoint_path)
        if last_check:
            print("Found checkpoint. Loading model.")
            checkpoint_id = int(os.path.basename(last_check).split('.')[2].split('-')[0]) #May need fine tuning
            print("Epoch: " + str(checkpoint_id) + "\n Resuming from: " + os.path.basename(last_check))
            utils_dict = {'decoding_L1_loss': L1_loss,
                          'decoding_L2_loss': L2_loss,
                          'encoding_deviation_loss': dev_loss,
                          'bit_error_rate': BER,
                          'median_encoding_deviation': med_dev,
                          'max_encoding_deviation': max_dev}
            training_model = keras.models.load_model(last_check,
                                                     custom_objects=utils_dict,
                                                     compile=False)  # Not sure if true or false
            args.lr = keras.backend.eval(training_model.optimizer.lr)
        else:
            print("No checkpoint found.")
            checkpoint_id = 0
            raise AttributeError
    else:
        print("Starting new training, with no prior model checkpoint.")
        checkpoint_id = 0
        dense_nums = tuple(args.dense_nums)
        training_model = get_training_model(info_len_k=args.info_len,
                                            codeword_len_n=args.codeword_len,
                                            stddev=args.noise_std,
                                            dense_nums=dense_nums)

    model_json = training_model.to_json
    json_path = os.path.join(args.checkpoint_path)
    with open(json_path, "w") as jf:
        jf.write(str(model_json))
        print("model json saved at " + json_path)

    info_gen =  InfoGenerator(info_len=args.info_len,
                              batch_size=args.batch_size,
                              shuffle=True)
    print("Testing Generator:")
    sample_info = info_gen[len(info_gen) - 1]
    print('Done')
    use_multiprocessing = args.num_generator_procs > 1

    metrics = {}
    metrics['encoder_output'] = [med_dev, max_dev]
    metrics['decoder_output'] = [BER]
    losses = [dev_loss, L1_loss, L2_loss]
    losses_weights = [0.33, 0.67, 0.0]
    optimizer = keras.optimizers.Adam(lr=args.lr)
    training_model.compile(loss=losses[0:2],
                           loss_weights=losses_weights[0:2],
                           optimizer=optimizer,
                           metrics=metrics)

    training_model.summary()
    callbacks = train_callbacks(log_path=args.checkpoint_path, checkpoint_path=args.checkpoint_path)
    training_model.fit_generator(generator=info_gen,
                                 max_queue_size=args.num_generator_procs*4, workers=args.num_generator_procs,
                                 use_multiprocessing=use_multiprocessing,
                                 epochs=args.epoch_num,
                                 verbose=1,
                                 initial_epoch=checkpoint_id,
                                 callbacks=callbacks)

if __name__ == '__main__':
    main(sys.argv[1:])





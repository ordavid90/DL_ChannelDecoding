from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, GaussianNoise

"""
probably better to re-write as standalone models, and then make composition. that way we can control the outputs coming
out of each sub-model (for instance the decoded before and after shrinking, and adjust losses appropriately).
"""
def get_encoded(input_tensors, in_size, out_size, dense_num):
    inputs = input_tensors
    x = Dense(0.5*(in_size + out_size), activation='relu')(inputs)
    for i in range(dense_num-2):
        x = Dense(out_size, activation='relu')(x)
    x = Dense(out_size, activation='tanh', name='encoder_output')(x)
    return x


def add_gaussian_noise(input_tensors, stddev):
    inputs = input_tensors
    x = GaussianNoise(stddev)(inputs)
    return x


def get_decoded(input_tensors, in_size, out_size, dense_num):
    inputs = input_tensors
    x = Dense(in_size, activation='relu')(inputs)
    for i in range(dense_num-2):
        x = Dense(in_size, activation='relu')(x)
    x = Dense(0.5*(in_size + out_size), activation='relu')(inputs)
    x = Dense(out_size, activation='tanh', name='decoder_output')(x)
    return x


def get_training_model(info_len_k=800, codeword_len_n=1024, stddev=1.0, dense_nums=(3, 3)):
    inputs = Input((info_len_k,), name='info_input')
    encoded = get_encoded(inputs, in_size=info_len_k, out_size=codeword_len_n, dense_num=dense_nums[0])
    noisy = add_gaussian_noise(encoded, stddev)
    decoded = get_decoded(noisy, in_size=codeword_len_n, out_size=codeword_len_n, dense_num=dense_nums[1])
    outputs = [encoded, decoded]
    training_model = Model(inputs, outputs, name=('NL_a_'+'_'.join([str(c) for c in dense_nums])))
    return training_model










#
#
# def initial_word_gen(k, num_words, binary=True):
#     if (binary):  # creating binary words
#         words_out = np.random.randint(2, size=(num_words, k))
#
#     else:
#         words_out = np.random.rand(num_words, k)
#         words_out = (words_out * 2) - 1
#
#     return words_out


# k = 800
# n = 1024
# batch_size = 32
# words = batch_size * 1000
# epoch_num = 5
#
# input = keras.layers.Input(shape=(k,))
# enc1    = keras.layers.Dense(n, activation=('hard_sigmoid'))(input)
# noise   = keras.layers.GaussianNoise(2)(enc1)
# dec1    = keras.layers.Dense(n, activation=('softmax'))(noise)
#
# model   = keras.Model(input=input, output=dec1)
#
# model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
#               # Loss function to minimize
#               loss=keras.losses.MeanSquaredError,
#               # List of metrics to monitor
#               metrics=['Accuracy'])
#
# enc_output  = model.get_layer(enc1).output
# input_words = initial_word_gen(k, words)
#
# model.fit(input_words, enc_output, batch_size=batch_size, epochs=epoch_num)

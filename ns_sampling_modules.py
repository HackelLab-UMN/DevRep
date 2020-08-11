import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import ns_plot_modules as pm
tf.config.optimizer.set_jit(True)

def incorrect_blanks(random_AA, random_AA_pos, seq):
    'return a logical array , where true means an invalid sample'
    # make sure that its in position 3,4,11,12
    # and if its in position 4,
    # make sure 3 is 19 otherwise resample
    # and if its in position 12,
    # make sure 11 is 19 otherwise resample
    out_of_bounds_regoin = np.bitwise_and(random_AA == 19,
                                          np.bitwise_and(
                                              np.bitwise_and(random_AA_pos != 4, random_AA_pos != 3),
                                              np.bitwise_and(random_AA_pos != 11, random_AA_pos != 12)
                                          ))
    # if the AA at position 3/11 wants to change to AA 19, then make sure that AA at position 4/12 is 19
    invalid_blank_regoin = np.bitwise_and(random_AA == 19,
                                          np.bitwise_or(
                                              np.bitwise_and(random_AA_pos == 3, seq[:, 4] != 19),
                                              np.bitwise_and(random_AA_pos == 11, seq[:, 12] != 19)
                                          ))
    # if the AA at position 4/12 wants to change to something other than 19 and 3/11 is currently 19,
    # then that is an invalid move
    invalid_change_back = np.bitwise_and(random_AA != 19,
                                         np.bitwise_or(
                                             np.bitwise_and(random_AA_pos == 4, seq[:, 3] == 19),
                                             np.bitwise_and(random_AA_pos == 12, seq[:, 11] == 19)
                                         ))

    change_blanks = np.bitwise_or(np.bitwise_or(out_of_bounds_regoin, invalid_blank_regoin), invalid_change_back)
    return change_blanks


def remove_blanks(random_AA_pos, random_AA, seq, generator):
    'removes sequences which are out of sequence space'
    # seq is a numpy 2D array of ordinals.
    # random_AA_pos is the random position of Amino acids: this does not change in these functions
    # random_AA is random AA to change to.

    change_blanks = incorrect_blanks(random_AA=random_AA, random_AA_pos=random_AA_pos, seq=seq)
    size=[-1]
    while change_blanks.any():
        #TODO: to deal with if your going to change position 5 and its 19 and 4 is 19 , 5 must be
        # resampled to 19...
        size.append(np.count_nonzero(change_blanks))
        print('change these blanks %i' % size[-1])
        random_AA[change_blanks] = sample(nb_of_sequences= size[-1],Nb_positions= 0, generator=generator)
        change_blanks = incorrect_blanks(random_AA=random_AA, random_AA_pos=random_AA_pos, seq=seq)
    return random_AA


def convert2pandas(ordinals_np):
    ord_pd = []
    # make a list of tuples
    for i in ordinals_np:
        ord_pd.append((i))
    return ord_pd


def convert2numpy(df, field='Ordinal'):
    return np.copy(np.array(df[field].to_numpy().tolist()))


def sample(nb_of_sequences, Nb_positions, generator,minval=0,maxval=21):
    'if nb_positions is 0 then will '
    if Nb_positions is 0:
        return generator.uniform(shape=[nb_of_sequences], minval=minval, maxval=maxval, dtype=tf.int64).numpy()
    return generator.uniform(shape=[nb_of_sequences, Nb_positions], minval=minval, maxval=maxval, dtype=tf.int64).numpy()
    # have two generators


def make_sampling_data(generator, Nb_sequences=1000, Nb_positions=16):
    # TODO: make two different generators to speed up process, but for now i think this will do ...
    'make sampling data and then remove all the blanks'
    seq = sample(nb_of_sequences=Nb_sequences, Nb_positions=Nb_positions, generator=generator)
    for k in np.arange(Nb_positions):
        # at the kth position in every sequence
        seq[:, k] = remove_blanks(random_AA_pos=np.ones((Nb_sequences)) * k, random_AA=seq[:, k].copy(), seq=seq,
                                  generator=generator)
    return convert2pandas(seq)


def _unit_tests_accuracy_blank_removal():
    'this is a unit test for removal of blanks, not for actual usage.'
    seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
    g_parent = tf.random.experimental.Generator.from_seed(seed_parent)
    seq = np.array([[0, 0, 0, 19, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 19, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    random_AA = np.array([4, 19])
    random_AA_pos = np.array([4, 12])
    remove_blanks(random_AA=random_AA, random_AA_pos=random_AA_pos, seq=seq, generator=g_parent)
from scripts.script_test_secML_attack_on_Keras_model import attack_keras_model, transform_dataset
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from scripts.gradient_reversal import GradientReversalModel
from scripts.bayesian_model import BayesianModel as bm
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import logging

logger = logging.getLogger(__name__)

# logger.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.INFO)

GRADIENT_REVERSAL_LAMBDA = 100
BATCH_SIZE = 512
MAX_EPOCHS = 15


def plot_history(history, save=None):
    """
    Small utility function to plot keras' history object.

    :param history: the keras history object
    :param save: either None (default) or a path to save the training object to.
    :return: No return values.
    """
    plt.figure(figsize=(10, 7.5))

    # summarize history for accuracy

    plt.subplot(311)
    plt.plot(history.history['output_mean_squared_error'])
    # plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(312)
    plt.plot(history.history['output2_acc'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(313)
    plt.plot(history.history['output_loss'])
    plt.plot(history.history['output2_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Y', 'A', 'combined'], loc='upper left')

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def main():
    df = pd.read_csv(os.path.join("..", "data", "csv", "scikit",
                                  "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))

    df_binary, Y, S = transform_dataset(df)

    df_outcome = pd.DataFrame(np.array([S, Y]).T, columns=['race', 'true'])

    logger.info("creating a new model with {} inputs".format(df_binary.shape[1]))

    gm = GradientReversalModel(GRADIENT_REVERSAL_LAMBDA, input_shape=df_binary.shape[1])
    history = gm.get_model().fit(df_binary,
                                 {"output": Y,
                                  "output2": preprocessing.OneHotEncoder().fit_transform(
                                      np.array(S).reshape(-1, 1)).toarray()},
                                 epochs=MAX_EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 verbose=0)

    plot_history(history, save="base.png")
    evolution = []

    df_outcome['pred'] = gm.predict(df_binary)[0]
    dem_parity = abs(
        bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian") - bm(df_outcome).P(pred=lambda x: x > 4).given(
            race="African-American"))

    eq_op = abs(bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian", true=True) - bm(df_outcome).P(
        pred=lambda x: x > 4).given(race="African-American", true=True))

    evolution.append({'DP': dem_parity, 'EO': eq_op})

    # Set up an interator with console progress
    t_prog = trange(5, desc='Progress', leave=True)

    for i in t_prog:
        try:
            result_pts, result_class = attack_keras_model(
                df_binary,
                Y=Y,
                S=S,
                nb_attack=1)

            attacked_df = pd.concat([
                df_binary,
                pd.DataFrame(result_pts, columns=df_binary.columns)])

            S_clean = preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray()
            S_new = np.random.randint(2, size=len(result_class))

            logger.info("creating a new model with {} inputs".format(df_binary.shape[1]))

            gm = GradientReversalModel(100, input_shape=attacked_df.shape[1])
            history = gm.get_model().fit(attacked_df,
                                         {"output": np.concatenate([Y, result_class[:, 0]]),
                                          "output2": np.concatenate([S_clean, np.array([S_new, 1 - S_new]).T])},
                                         epochs=MAX_EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         verbose=0)

            plot_history(history, save="{}.png".format(i))

            df_outcome['pred'] = gm.predict(df_binary)[0]

            dem_parity = abs(
                bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian") - bm(df_outcome).P(
                    pred=lambda x: x > 4).given(
                    race="African-American"))
            logger.info("Bias: {}".format(dem_parity))

            eq_op = abs(bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian", true=True) - bm(df_outcome).P(
                pred=lambda x: x > 4).given(race="African-American", true=True))
            logger.info("Bias: {}".format(eq_op))

            evolution.append({'DP': dem_parity, 'EO': eq_op})

            t_prog.set_postfix(evolution[-1])  # print last metrics
            t_prog.refresh()

        except IndexError:
            pass


if __name__ == '__main__':
    main()

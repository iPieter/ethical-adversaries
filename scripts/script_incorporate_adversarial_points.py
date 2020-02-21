from scripts.script_test_secML_attack_on_Keras_model import attack_keras_model, transform_dataset
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from scripts.gradient_reversal import GradientReversalModel
from scripts.bayesian_model import BayesianModel as bm
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.INFO)


def plot_history(history):
    "Small utility function to plot keras' history object."

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
    plt.show()


def main():
    df = pd.read_csv(os.path.join("..", "data", "csv", "scikit",
                                  "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))

    df_binary, Y, S = transform_dataset(df)

    df_outcome = pd.DataFrame(np.array([S, Y]).T, columns=['race', 'true'])

    logging.info("creating a new model with {} inputs".format(df_binary.shape[1]))

    GRADIENT_REVERSAL_LAMBDA = 100
    BATCH_SIZE = 32
    MAX_EPOCHS = 1500

    gm = GradientReversalModel(GRADIENT_REVERSAL_LAMBDA, input_shape=df_binary.shape[1])
    history = gm.get_model().fit(df_binary,
                                 {"output": Y,
                                  "output2": preprocessing.OneHotEncoder().fit_transform(
                                      np.array(S).reshape(-1, 1)).toarray()},
                                 epochs=MAX_EPOCHS,
                                 batch_size=BATCH_SIZE)

    plot_history(history)

    df_outcome['pred'] = gm.predict(df_binary)[0]
    result_pts, result_class = attack_keras_model(df_binary, Y=Y, S=S)

    dem_parity = abs(
        bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian") - bm(df_outcome).P(pred=lambda x: x > 4).given(
            race="African-American"))
    logging.info("Bias: {}".format(dem_parity))

    eq_op = abs(bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian", true=True) - bm(df_outcome).P(
        pred=lambda x: x > 4).given(race="African-American", true=True))
    logging.info("Bias: {}".format(eq_op))

    attacked_df = pd.concat([df_binary, pd.DataFrame(result_pts, columns=df_binary.columns)])

    S_clean = preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray()
    S_new = np.random.randint(2, size=len(result_class))

    logging.info("creating a new model with {} inputs".format(df_binary.shape[1]))

    gm = GradientReversalModel(100, input_shape=attacked_df.shape[1])
    history = gm.get_model().fit(attacked_df,
                       {"output": np.concatenate([Y, result_class[:, 0]]),
                        "output2": np.concatenate([S_clean, np.array([S_new, 1 - S_new]).T])},
                       epochs=MAX_EPOCHS,
                       batch_size=BATCH_SIZE)

    plot_history(history)

    df_outcome['pred'] = gm.predict(df_binary)[0]

    dem_parity = abs(
        bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian") - bm(df_outcome).P(pred=lambda x: x > 4).given(
            race="African-American"))
    logging.info("Bias: {}".format(dem_parity))

    eq_op = abs(bm(df_outcome).P(pred=lambda x: x > 4).given(race="Caucasian", true=True) - bm(df_outcome).P(
        pred=lambda x: x > 4).given(race="African-American", true=True))
    logging.info("Bias: {}".format(eq_op))


if __name__ == '__main__':
    main()

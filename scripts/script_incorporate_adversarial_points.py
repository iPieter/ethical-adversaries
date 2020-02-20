from scripts.script_test_secML_attack_on_Keras_model import attack_keras_model, transform_dataset
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from scripts.gradient_reversal import GradientReversalModel
from scripts.bayesian_model import BayesianModel as bm
import logging
logging.basicConfig(format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s'", level=logging.INFO)


def main():
    df = pd.read_csv(os.path.join("..", "data", "csv", "scikit", "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))

    df_binary, Y, S = transform_dataset(df)

    df_outcome = pd.DataFrame(np.array([S,Y]).T, columns=['race', 'true'])

    logging.info("creating a new model with {} inputs".format(df_binary.shape[1]))

    gm = GradientReversalModel(100, input_shape=df_binary.shape[1])
    gm.get_model().fit(df_binary,
                       {"output": Y,
                        "output2": preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray()},
                       epochs=500,
                       batch_size=32)

    df_outcome['pred'] = gm.predict(df_binary)[0]

    result_pts, result_class = attack_keras_model(df_binary, Y=Y, S=S)

    print(pd.DataFrame(result_pts).shape)

    print(df_binary.shape)

    attacked_df = pd.concat([df_binary, pd.DataFrame(result_pts, columns=df_binary.columns)])



if __name__ == '__main__':
    main()
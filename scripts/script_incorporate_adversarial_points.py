from scripts.script_test_secML_attack_on_Keras_model import attack_keras_model
import pandas as pd
import os
from sklearn import preprocessing
import numpy as np

def transform_dataset(df):
    """

    :param df:
    :return: Tuple of the transformed dataset and the labels Y and S
    """

    df_binary = df[(df["race"] == "Caucasian") | (df["race"] == "African-American")]

    del df_binary['c_jail_in']
    del df_binary['c_jail_out']

    ##separated class from the rests of the features
    # remove unnecessary dimensions from Y -> only the decile_score remains
    Y_to_remove = df_binary['decile_score']
    del df_binary['decile_score']
    del df_binary['two_year_recid']
    del df_binary['score_text']

    S = df_binary['race']
    del df_binary['race']

    # del X[nb_feat-1]
    Y = df_binary['is_recid']
    df_binary.drop(columns='is_recid', inplace=True)
    print(df_binary.head())

    # set sparse to False to return dense matrix after transformation and keep all dimensions homogeneous
    encod = preprocessing.OneHotEncoder(sparse=False)

    data_to_encode = df_binary.to_numpy()
    feat_to_encode = data_to_encode[:, 0]
    # print(feat_to_encode)
    # transposition
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    # print(feat_to_encode)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_binary_encoded = pd.DataFrame(encoded_feature)

    df_binary_encoded.head()

    feat_to_encode = data_to_encode[:, 1]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_encoded_feature = pd.DataFrame(encoded_feature)

    df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    # feature [2] [3] [4] [5] [6] [7] [8] has to be put between 0 and 1

    for i in range(2, 9):
        encoded_feature = data_to_encode[:, i]
        ma = np.amax(encoded_feature)
        mi = np.amin(encoded_feature)
        encoded_feature = (encoded_feature - mi) / (ma - mi)
        df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 9]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_encoded_feature = pd.DataFrame(encoded_feature)

    df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    return df_binary_encoded, Y, S

def main():
    df = pd.read_csv(os.path.join("..", "data", "csv", "scikit", "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))

    df, Y, S = transform_dataset(df)

    df_attacked = pd.DataFrame(attack_keras_model(df, Y=Y, S = S))
    df_attacked = df_attacked.rename({'14': 'label'})


if __name__ == '__main__':
    main()
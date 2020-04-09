import preprocessing as pp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def sample(df, m=1000, mots="1_ortho", phon="2_phon", occurances="10_freqlivres", ln_dist=False):
    """
    :param df: pd.dataframe contenant le lexique
    :param mots: "1_ortho" variable de df contenant les orthographes
    :param phon: "2_phon" variable de df contenant les phonemes
    :param occurances: "10_freqlivres" variable de df contenant les frequences des mots
    :param ln_dist: False passage au log
    :param m: 1000 taille des donnees

    :return: liste de tuples (mot, prononciation), liste contenant les probabilités
    """
    list_w2p = []
    list_occ = []
    for row in df[[mots, phon, occurances]].to_numpy():
        w, p, o = tuple(row)
        list_w2p.append([w, p])
        list_occ.append(o)
    list_occ = np.array(list_occ)

    # normalisation
    if ln_dist:
        list_occ = np.log(list_occ + 1)
    list_occ = list_occ / np.sum(list_occ)

    # format liste
    list_tuples = [tuple(couple) for couple in list_w2p]
    list_occ = list_occ.tolist()
    n_occ = len(list_tuples)
    distr = np.random.choice(a=range(n_occ), size=m, p=list_occ).tolist()
    return [list_tuples[i] for i in distr]


def train_dev(df, test_size=0.01, m=1000, mots="1_ortho", phon="2_phon", occurances="10_freqlivres", ln_dist=False):
    """
    :param df: pd.dataframe contenant le lexique
    :param test_size: 0.01
    :param m: 1000 taille des donnees de train
    :param mots: "1_ortho" variable de df contenant les orthographes
    :param phon: "2_phon" variable de df contenant les phonemes
    :param occurances: "10_freqlivres" variable de df contenant les frequences des mots
    :param ln_dist: False passage au log

    :return: list
    """
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_s = sample(train_df, m=m, mots=mots, phon=phon, occurances=occurances, ln_dist=ln_dist)
    test_l = [tuple(r) for r in test_df[[mots, phon]].to_numpy()]
    return train_s, test_l


def one_hot_from_list(data, tx, ty, l2idx, p2idx, blank="_"):
    """
    :param data:
    :param tx:
    :param ty:
    :param l2idx:
    :param p2idx:
    :param blank:

    :return:
    """
    m = len(data)
    n_l = len(l2idx.keys())
    n_p = len(p2idx.keys())
    x = np.zeros((m, tx + 1, n_l))
    y = np.zeros((m, ty + 1, n_p))
    for i, mp in enumerate(data):
        mot, pron = ("{m}{b}".format(m=mp[0], b=blank * (tx + 1 - len(mp[0]))),  # rajout des _ pour signifier la fin
                     "{m}{b}".format(m=mp[1], b=blank * (ty + 1 - len(mp[1]))))
        for j, c in enumerate(mot):
            x[i, j, l2idx[c]] = 1
        for j, c in enumerate(pron):
            y[i, j, p2idx[c]] = 1
    return x, y


# _, df_w2p = pp.set_ortho2phon(pp.import_lexique_as_df(), accent_e=True)
df_w2p = pd.DataFrame(data={"1_ortho": ["vache", "cheval", "fghui"],
                            "2_phon": ["vache", "cval", "fgh8ui"],
                            "10_freqlivres": [0.1, 8.4, 2.1]})
tr_l, ts_l = train_dev(df_w2p, m=800000, ln_dist=True)
ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
x_train, y_train = one_hot_from_list(tr_l, Tx, Ty, ltr2idx, phon2idx)

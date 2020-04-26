import preprocessing as pp
# import time
import numpy as np
from random import shuffle
from scipy.sparse import csr_matrix
import pandas as pd
from nltk.tag import StanfordPOSTagger
import keras as k
from sklearn.model_selection import train_test_split
# from tqdm import tqdm


def filter_length(df, n=8, phon="2_phon", sup_n=True):
    phonemes = df.loc[:, phon]
    if sup_n:
        filtre = phonemes.apply(lambda x: len(x) >= n)
    else:
        filtre = phonemes.apply(lambda x: len(x) < n)
    return df.loc[filtre, :]


def sample(df, m=1000, mots="1_ortho", phon="2_phon", occurances="10_freqlivres", ln_dist=False, seed=23):
    """
    :param df: pd.dataframe contenant le lexique
    :param mots: "1_ortho" variable de df contenant les orthographes
    :param phon: "2_phon" variable de df contenant les phonemes
    :param occurances: "10_freqlivres" variable de df contenant les frequences des mots
    :param ln_dist: False passage au log
    :param m: 1000 taille des donnees
    :param seed: graine aleatoire de l'echantillonage

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
    np.random.seed(seed)
    distr = np.random.choice(a=range(n_occ), size=m, p=list_occ).tolist()
    return [list_tuples[i] for i in distr]


def train_dev(df, test_size=0.01, m=1000, forced_train=None, mots="1_ortho", phon="2_phon",
              occurances="10_freqlivres", ln_dist=False, seed=23):
    """
    :param df: pd.dataframe contenant le lexique
    :param test_size: 0.01
    :param m: 1000 taille des donnees de train
    :param forced_train: liste de mots a avoir dans les donnees d'entrainement
    :param mots: "1_ortho" variable de df contenant les orthographes
    :param phon: "2_phon" variable de df contenant les phonemes
    :param occurances: "10_freqlivres" variable de df contenant les frequences des mots
    :param ln_dist: False passage au log
    :param seed: graine aleatoire du train_test_split et de l'echantillonage

    :return: listes de tuples des train
    """
    if forced_train is None:
        forced_train = []
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    if len(forced_train) > 0:  # rajout des mots dans les donnees de test
        forced_idx = test_df[mots].apply(lambda x: x in forced_train)
        forced = test_df.loc[forced_idx, :]
        train_df = train_df.append(forced, ignore_index=True)
        test_df = test_df.loc[-forced_idx, :]
    train_s = sample(train_df, m=m, mots=mots, phon=phon, occurances=occurances, ln_dist=ln_dist, seed=seed)
    test_s = sample(test_df, m=int(m * test_size), mots=mots, phon=phon, occurances=occurances,
                    ln_dist=ln_dist, seed=seed)
    return train_s, test_s


def model_test(tx, ty, n_l, n_p, n_brnn1=32, n_h1=64):
    x = k.Input(shape=(tx, n_l))
    c0 = k.Input(shape=(n_h1,), name='c0')
    h0 = k.Input(shape=(n_h1,), name='h0')
    c = c0
    h = h0
    outputs = list()  # initialisation de la derniere couche

    # c'est parti
    a = k.layers.Bidirectional(k.layers.LSTM(units=n_brnn1, return_sequences=True, name="LSTM_mot"))(x)
    a = k.layers.Dropout(0.2, name="dropout_LSTM_orthographe")(a)
    for t in range(ty):
        # Attention
        h_rep = k.layers.RepeatVector(tx, name="att_repeat_phoneme{}".format(t))(h)
        ah = k.layers.Concatenate(axis=-1, name="att_concat_phoneme{}".format(t))([h_rep, a])
        energies = k.layers.Dense(units=n_h1, activation="tanh", name="att_caractere_phoneme{}".format(t))(ah)
        energies = k.layers.Dense(units=1, activation="relu", name="att_moyenne_phoneme{}".format(t))(energies)
        alpha = k.layers.Activation("softmax", name="att_alpha_phoneme{}".format(t))(energies)
        context = k.layers.Dot(axes=1, name="att_application_phoneme{}".format(t))([alpha, a])

        h, c = k.layers.GRU(units=n_h1, activation='tanh', recurrent_activation='tanh', return_state=True,
                            name="LSTM_phoneme{}".format(t))(inputs=context, initial_state=c)
        h = k.layers.Dropout(rate=0.1, name="dropout_phoneme{}".format(t))(h)
        c = k.layers.Dropout(rate=0.1, name="dropout_memory_phoneme{}".format(t))(c)
        outy = k.layers.Dense(activation="softmax",
                              units=n_p, name="LSTM_{}".format(t))(h)
        outputs.append(outy)
    net = k.models.Model(inputs=[x, c0, h0], outputs=outputs)
    return net


def tokenizer(phrase):
    """
    Tokeniseur maison qui sépare autour des espaces et les appostophes des mots plus courts que deux chars

    :param phrase
    :return: phrase tokenisee
    """
    # gestion appostrophes
    splited_no_appostrophes = phrase.split("'")
    splited_with_appostrophes = ["{}'".format(splited_no_appostrophes[i])
                                 for i in range(len(splited_no_appostrophes) - 1)]
    if len(splited_no_appostrophes[-1]) > 0:
        splited_with_appostrophes.append(splited_no_appostrophes[-1])

    # gestion des espaces
    mots = list()
    for sans_appostrophe in splited_with_appostrophes:
        mots.extend(sans_appostrophe.split(" "))

    idx_mot = 0
    while idx_mot < len(mots) - 1:  # -1 pour ne pas aller sur le dernier mot
        if mots[idx_mot][-1] == "'" and len(mots[idx_mot]) > 3:
            mots[idx_mot] = "{}{}".format(mots[idx_mot], mots[idx_mot + 1])
            mots.pop(idx_mot + 1)
        else:
            idx_mot += 1
    return mots


def pos_tag(mots,
            jar=r"C:\Users\remif\PythonRequired\stanford-postagger-full-2017-06-09\stanford-postagger-3.8.0.jar",
            mdl=r"C:\Users\remif\PythonRequired\stanford-postagger-full-2017-06-09\models\french-ud.tagger"):
    pos_tagger = StanfordPOSTagger(mdl, jar, encoding='utf8')
    return pos_tagger.tag(mots)


def liaison(ortho1, ortho2, phon1, phon2):
    voyelles_p = ['a', 'E', '§', 'o', 'O', '1', 'i', '5', 'e', 'u', '@', '°', '9', 'y', '2']
    consonnes_p = ['k', 'p', 'l', 't', 'R', 'j', 'f', 's', 'd', 'Z', 'n', 'b', 'v', 'g',
                'v', 'g', 'm', 'z', 'w', 'S', 'N', '8', 'G', 'x']
    mot2_liable = (phon2[0] in voyelles_p) and (ortho2[0] != 'h')
    if mot2_liable:
        if ortho1[-1] in ['d']:
            phon1 = "{}t".format(phon1)
    return phon1, phon2


# for _, mot, phon in df_w2p.loc[:, ['1_ortho', '2_phon']].itertuples():
#     if phon[0] in consonnes_p:  # and phon[-1] in consonnes:
#         print("{} : {}".format(mot, phon))

class Lecteur:
    """"Classe definissant le lecteur
    """
    def __init__(self, tx, ty, l2idx, p2idx, n_brnn1=90, n_h1=80, net=None, blank="_"):
        self.tx = tx
        self.ty = ty
        self.l2idx = l2idx
        self.p2idx = p2idx
        self.n_brnn1 = n_brnn1
        self.n_h1 = n_h1
        self.net = net
        self.blank = blank

    def one_hot_from_list(self, data):
        """
        :param data:

        :return:
        """
        m = len(data)
        n_l = len(self.l2idx.keys())
        n_p = len(self.p2idx.keys())
        x = np.zeros((m, self.tx + 1, n_l))
        y = np.zeros((m, self.ty + 1, n_p))
        for i, mp in enumerate(data):
            mot, pron = ("{m}{b}".format(m=mp[0], b=self.blank * (self.tx + 1 - len(mp[0]))),  # rajout des _ pour
                         # signifier la fin
                         "{m}{b}".format(m=mp[1], b=self.blank * (self.ty + 1 - len(mp[1]))))
            for j, c in enumerate(mot):
                x[i, j, self.l2idx[c]] = 1
            for j, c in enumerate(pron):
                y[i, j, self.p2idx[c]] = 1
        return x, y

    # Modelisation
    def model(self):
        n_l = len(self.l2idx)
        n_p = len(self.p2idx)
        x = k.Input(shape=(self.tx, n_l), name="mot")
        c0 = k.Input(shape=(self.n_h1,), name='c0')
        c = c0
        h0 = k.Input(shape=(self.n_h1,), name='h0')
        h = h0
        outputs = list()  # initialisation de la derniere couche

        # c'est parti
        a = k.layers.Bidirectional(k.layers.LSTM(units=self.n_brnn1, return_sequences=True, name="LSTM_orthographe"))(x)
        a = k.layers.Dropout(0.2, name="dropout_LSTM_orthographe")(a)
        for t in range(self.ty):
            # Attention
            h_rep = k.layers.RepeatVector(self.tx, name="att_repeat_phoneme{}".format(t))(h)
            ah = k.layers.Concatenate(axis=-1, name="att_concat_phoneme{}".format(t))([h_rep, a])
            energies = k.layers.Dense(units=self.n_h1, activation="tanh", name="att_caractere_phoneme{}".format(t))(ah)
            energies = k.layers.Dense(units=1, activation="relu", name="att_moyenne_phoneme{}".format(t))(energies)
            alpha = k.layers.Activation("softmax", name="att_alpha_phoneme{}".format(t))(energies)
            context = k.layers.Dot(axes=1, name="att_application_phoneme{}".format(t))([alpha, a])

            h, c = k.layers.GRU(units=self.n_h1, activation='tanh', recurrent_activation='sigmoid', return_state=True,
                                name="GRU_phoneme{}".format(t))(inputs=context, initial_state=c)

            h = k.layers.Dropout(rate=0.1, name="dropout_phoneme{}".format(t))(h)
            c = k.layers.Dropout(rate=0.1, name="dropout_memory_phoneme{}".format(t))(c)

            outy = k.layers.Dense(activation="softmax",
                                  units=n_p, name="softmax_phoneme_{}".format(t))(h)
            outputs.append(outy)
        net = k.models.Model(inputs=[x, c0, h0], outputs=outputs)
        return net

    def compile_train(self, x, y, epochs=10, batch_size=64, opt=k.optimizers.Adam()):
        m = x.shape[0]
        if self.net is None:
            self.net = self.model()
            self.net.summary()
        self.net.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        h0 = csr_matrix((m, self.n_h1))
        c0 = csr_matrix((m, self.n_h1))
        y_list = list(y.swapaxes(0, 1))
        self.net.fit([x, c0, h0], y_list, epochs=epochs, batch_size=batch_size)
        return self.net

    def lire(self, mot):
        mot = str(mot)
        mot = "{m}{b}".format(m=mot, b=self.blank * (self.tx + 1 - len(mot)))
        lettres = [self.l2idx[lettre] for lettre in mot]
        n_l = len(self.l2idx.keys())
        x = np.zeros((1, self.tx + 1, n_l))
        h0 = csr_matrix((1, self.n_h1))
        c0 = csr_matrix((1, self.n_h1))
        for i, lettre in enumerate(lettres):
            x[0, i, lettre] = 1
        y = self.net.predict(x=[x, c0, h0])
        idx2p = dict()
        for p, idx in self.p2idx.items():
            idx2p[idx] = p
        pron_int = np.concatenate(y).argmax(axis=1).tolist()
        prononciation = ""
        for idx in pron_int:
            prononciation = "{phons}{p}".format(phons=prononciation, p=idx2p[idx]).replace(self.blank, "")
        return prononciation

    def lire_phrase(self, phrase):
        # tokenisation
        mots = tokenizer(phrase)

        # one hot
        m = len(mots)
        n_l = len(self.l2idx)
        x = np.zeros((m, self.tx + 1, n_l))
        for i, mot in enumerate(mots):
            mot = str(mot)
            mot = "{m}{b}".format(m=mot, b=self.blank * (self.tx + 1 - len(mot)))
            lettres = [self.l2idx[lettre] for lettre in mot]
            for j, lettre in enumerate(lettres):
                x[i, j, lettre] = 1

        # predictions
        h0 = csr_matrix((m, self.n_h1))
        c0 = csr_matrix((m, self.n_h1))
        y = self.net.predict(x=[x, c0, h0])

        # dictionnaire indices vers phoneme
        idx2p = dict()
        for p, idx in self.p2idx.items():
            idx2p[idx] = p

        # conversion one hot vers mots
        n_p = len(self.p2idx)
        pron_int = np.zeros((m, n_p, self.ty + 1))
        for i in range(self.ty + 1):
            pron_int[:, :, i] = y[i]
        pron_int = pron_int.argmax(axis=1)  # one hot vers indices phonemes

        prononciation = ""
        for i in range(m):
            if i > 0:
                prononciation += " "  # peut-etre a enlever apres le scrapping
            prononciation_mot = ""
            for idx in pron_int[i, :].tolist():
                prononciation_mot = "{phons}{p}".format(phons=prononciation_mot,
                                                        p=idx2p[idx]).replace(self.blank, "")
            prononciation += prononciation_mot
        return prononciation

    def evaluate_model_from_lists(self, liste, batch_size=256):
        """
        Mesure la performance d'un modele sur une liste de donnees
        :param liste: donnees sous forme de liste sur lesquelles evaluer le modele
        :param batch_size: taille du batch pour calculer les performances
        """
        x, y_inv = self.one_hot_from_list(liste)
        y = list(y_inv.swapaxes(0, 1))
        h0 = csr_matrix((x.shape[0], self.n_h1))
        c0 = csr_matrix((x.shape[0], self.n_h1))
        results = self.net.evaluate([x, c0, h0], y, batch_size=batch_size)
        print('loss:', results[0])
        for i in range(0, self.ty):
            print('acc char {}:'.format(i), results[i + self.ty + 2])

    def mispredicted(self, t_l, batch_size=256):
        unique_couples = list(set(t_l))  # recuperation des elements uniques
        missed = list()  # liste des éléments mal predits
        x, y = self.one_hot_from_list(data=unique_couples)
        m = x.shape[0]
        y_a = y.argmax(axis=2)
        print(y_a.shape)
        c0 = csr_matrix((m, self.n_h1))
        h0 = csr_matrix((m, self.n_h1))
        y_hat_list = self.net.predict(x=[x, c0, h0], batch_size=batch_size)
        pred_phon = list()

        # dictionnaire indice vers phoneme
        idx2p = dict()
        for p, idx in self.p2idx.items():
            idx2p[idx] = p

        for i in range(self.ty + 1):
            mat_yi_hat = y_hat_list[i].argmax(axis=1).reshape((m, 1))
            pred_phon.append(mat_yi_hat)
        y_hat_a = np.concatenate(pred_phon, axis=1)

        # tranformation des mauvaises predictions en mots
        for i in range(m):
            bonne_prediction = pd.Series(y_hat_a[i, :] == y_a[i, :]).all()
            if not bonne_prediction:
                vecteur_predit = y_hat_a[i, :].tolist()
                prononciation_predite = "".join([idx2p[idx] for idx in vecteur_predit]).replace(self.blank, "")
                missed.append([unique_couples[i][0], unique_couples[i][1], prononciation_predite])
        return missed

    def save(self, path):
        self.net.save(path)


_, df_w2p = pp.set_ortho2phon(pp.import_lexique_as_df(), accent_e=False)
# df_w2p = pd.DataFrame(data={"1_ortho": ["vache", "cheval", "fghui"],
#                             "2_phon": ["vache", "cval", "fgh8ui"],
#                             "10_freqlivres": [0.1, 8.4, 2.1]})
# _, ts_l = train_dev(df_w2p, m=1600000, forced_train=[r"où"], ln_dist=True)
df_inf8 = filter_length(df_w2p, n=10, sup_n=False)
tr_l, _ = train_dev(df_inf8, m=400000, forced_train=[r"où"], ln_dist=True)
df_sup8 = filter_length(df_w2p, n=10, sup_n=True)
tr_l_sup8, _ = train_dev(df_sup8, m=400000, forced_train=[r"où"], ln_dist=True)
tr_l.extend(tr_l_sup8)
# ts_l.extend(ts_l_sup8)
shuffle(tr_l)
# shuffle(ts_l)

ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
mdl = k.models.load_model(r"C:\Users\remif\PycharmProjects\PoemesProfonds\CE1_T11_redouble.h5")
lecteur = Lecteur(Tx, Ty, ltr2idx, phon2idx, n_brnn1=90, n_h1=80, net=mdl, blank="_")

x_train, y_train = lecteur.one_hot_from_list(tr_l)

# clone = model(Tx + 1, Ty + 1, len(ltr2idx), len(phon2idx), n_brnn1=90, n_h1=80)
# clone.summary()
#
# clone.load_weights(r"C:\Users\remif\PycharmProjects\PoemesProfonds\CE1_T5.h5")

lecteur.compile_train(x_train, y_train, epochs=1, batch_size=64,
                      opt=k.optimizers.Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.999))

lecteur.save(r"C:\Users\remif\PycharmProjects\PoemesProfonds\CE1_T12_l10.h5")

# evaluate_model_from_lists(ts_l, Tx, Ty, ltr2idx, phon2idx, lecteur, batch_size=256, n_h1=80)
# print(lire("montmartre", lecteur, ltr2idx, phon2idx, Tx, blank="_"))
#
# miss = mispredicted(ts_l, lecteur, Tx, Ty, ltr2idx, phon2idx, n_h1=80, batch_size=128)

# from tensorflow.python.client import device_lib
#
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

# clone = k.models.clone_model(lecteur)
# for i, layer in enumerate(clone.layers):
#     if type(layer) is k.layers.Dropout:
#         clone.layers[i].rate = 0.1
# clone.compile(optimizer=k.optimizers.Adam(learning_rate=0.0000025, beta_1=0.9, decay=0.999),
#               loss="categorical_crossentropy", metrics=["accuracy"])
# clone.load_weights(".\CP_GPU5.h5")

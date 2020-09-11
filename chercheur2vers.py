from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import Input
from keras.models import Model, Sequential
from keras.layers import LSTM, TimeDistributed, Dropout, BatchNormalization, Concatenate, Dense, Lambda, GRU, LeakyReLU


def model_test(t_p=51, n_p=38, dim_emb=300, n_rnnphonvers=300, n_rnn_phon15=420,
               n_dense_particulier=50, n_dense_particulier2=30, n_dense1=25):
    # phonemes
    phonemes_in = Input(shape=(9, t_p, n_p), name="phonemes_input")

    rnn_phon = GRU(units=n_rnnphonvers, activation="tanh", name="embedding_phonemes")

    phonems_v15 = TimeDistributed(rnn_phon, name="RNN_phonemes_vers_unique")(phonemes_in)
    phonems_v14 = Lambda(lambda x: x[:, :-1, :], output_shape=(8, n_rnnphonvers), name="phonemes_v")(phonems_v15)
    phonems_cible = Lambda(lambda x: x[:, -1, :], output_shape=(n_rnnphonvers,), name="phonemes_cible")(phonems_v15)

    rnn_phon_vers = LSTM(units=n_rnn_phon15, activation="tanh", name="RNN_phonemes_total")(phonems_v14)

    phonemes_tout = Concatenate(axis=1, name="concat_phonemes")([rnn_phon_vers, phonems_cible])
    phonemes_tout = Dense(units=n_dense_particulier, name="dense_phonemes")(phonemes_tout)
    phonemes_tout = LeakyReLU(0.2, name="activation_dense_phonemes")(phonemes_tout)
    phonemes_tout = Dropout(rate=0.1, name="dropout_phonemes")(phonemes_tout)
    phonemes_tout = BatchNormalization(name="batchnorm_phonemes")(phonemes_tout)
    phonemes_tout2 = Dense(units=n_dense_particulier2, name="dense_phonemes2")(phonemes_tout)
    phonemes_tout2 = LeakyReLU(0.2, name="activation_dense_phonemes2")(phonemes_tout2)
    phonemes_tout2 = Dropout(rate=0.1, name="dropout_phonemes2")(phonemes_tout2)
    phonemes_tout2 = BatchNormalization(name="batchnorm_phonemes2")(phonemes_tout2)

    # word embedding
    words_in = Input(shape=(9, dim_emb), name="words_input")

    emb_reduc = Sequential(name="dim_reduction")
    emb_reduc.add(Dense(units=n_rnnphonvers, activation="tanh"))
    emb_reduc.add(Lambda(lambda t: K.l2_normalize(1000*t, axis=1)))
    emb_reduc.add(Dropout(rate=0.1))
    emb_reduc.add(BatchNormalization())

    words_v15 = TimeDistributed(emb_reduc)(words_in)
    words_v14 = Lambda(lambda x: x[:, :-1, :], output_shape=(8, n_rnnphonvers), name="words_v")(words_v15)
    wordscible = Lambda(lambda x: x[:, -1, :], output_shape=(n_rnnphonvers,), name="wordscible")(words_v15)

    rnn_words = GRU(units=n_rnn_phon15, activation="tanh", name="RNN_words")(words_v14)

    words_tout = Concatenate(axis=1, name="concat_words")([rnn_words, wordscible])
    words_tout = Dense(units=n_dense_particulier, name="dense_words")(words_tout)
    words_tout = LeakyReLU(0.2, name="activation_dense_words")(words_tout)
    words_tout = Dropout(rate=0.1, name="dropout_words")(words_tout)
    words_tout = BatchNormalization(name="batchnorm_words")(words_tout)
    words_tout2 = Dense(units=n_dense_particulier2, name="dense_words2")(words_tout)
    words_tout2 = LeakyReLU(0.2, name="activation_dense_words2")(words_tout2)
    words_tout2 = Dropout(rate=0.1, name="dropout_words2")(words_tout2)
    words_tout2 = BatchNormalization(name="batchnorm_words2")(words_tout2)

    # rassemblement
    tout = Concatenate(axis=1, name="concat_tout")([phonemes_tout2, words_tout2])
    dense1 = Dense(units=n_dense1, name="dense_tout1")(tout)
    dense1 = LeakyReLU(0.2, name="activation_dense_tout1")(dense1)
    dense1 = Dropout(rate=0.1, name="dropout_tout1")(dense1)
    dense1 = BatchNormalization(name="batchnorm_tout1")(dense1)
    proba = Dense(units=1, activation="sigmoid", name="probability")(dense1)

    net = Model(inputs=[phonemes_in, words_in], outputs=proba)
    return net


def split_train_dev(df, test_size=0.02, var_id="id", seed=23, forced_id=None):
    if forced_id is None:
        forced_id = []
    id_series = df.loc[:, var_id]
    ids = list(set(id_series.to_list()))
    ids_train, ids_test = train_test_split(ids, test_size=test_size, random_state=seed)
    if len(forced_id) > 0:  # rajout des mots dans les donnees de test
        for id_to_learn_on in forced_id:
            if id_to_learn_on in ids_test:
                ids_test.remove(id_to_learn_on)
                ids_train.append(id_to_learn_on)
    bool_train_series = id_series.apply(lambda x: x in ids_train)
    df_train = df.loc[bool_train_series, :]
    df_test = df.loc[~bool_train_series, :]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_test


def fake_surroundings(len_poem, size_surroundings=5):
    """
    Retourne une liste d'indices tirée au sort
    :param len_poem: nombre de vers dans le poème
    :param size_surroundings: distance du vers de référence du vers (default 5)
    :return: liste
    """
    # bornes inférieures
    lower_bounds_w_neg = np.array([row - size_surroundings for row in range(len_poem)])
    lower_bounds_2d = np.stack([np.zeros(len_poem), lower_bounds_w_neg])
    # calcul max entre 0 et le rang - surroundings
    lower_bounds = np.max(lower_bounds_2d, axis=0)

    # bornes supérieures
    higher_bounds_w_neg = np.array([row + size_surroundings for row in range(len_poem)])
    higher_bounds_2d = np.stack([np.repeat(len_poem, len_poem), higher_bounds_w_neg])
    # calcul min entre longueur du poeme et le rang + surroundings
    higher_bounds = np.min(higher_bounds_2d, axis=0)

    # tirage
    fake_within_poem = np.random.randint(low=lower_bounds, high=higher_bounds).tolist()
    return fake_within_poem


def vers2phon_vect(vers, lecteur, ft):
    """
    Permet d'obtenir les phonèmes et le vecteur associé à un vers que l'on tape
    :param vers: string correspondant à un vers
    :param lecteur: instance de Lecteur qui permet de lire le vers
    :param ft: modèle FastText dans lequel se projète les vecteurs de mots
    :return: tuple phonemes, vecteur
    """
    if type(vers) is not str:
        raise TypeError
    phonemes = lecteur.lire_vers(vers, count=False)
    if len(vers) > 0:
        vect = ft.get_sentence_vector(vers)
        norm_vect = np.linalg.norm(vect)
        if norm_vect > 0:
            vect /= norm_vect
    else:
        vect = ft.get_sentence_vector("/s")
    return phonemes, vect


def pieds(vers):
    voyelles = "aE§oO1i5eu@°9y2"
    reg = "[{}]".format(voyelles)
    voyelles_vers = RegexpTokenizer(reg).tokenize(vers)
    nb_pieds = len(voyelles_vers)
    return nb_pieds


class Chercheur2Vers:
    def __init__(self, t_p, p2idx, dim_words=300, n_antecedant_vers=8, blank="_", net=None, var_id="id",
                 var_phonemes="phonemes", var_vects="vect", var_vers="vers"):
        self.t_p = t_p
        self.p2idx = p2idx
        self.dim_words = dim_words
        self.n_antecedant_vers = n_antecedant_vers
        self.blank = blank
        self.net = net
        self.var_id = var_id
        self.var_phonemes = var_phonemes
        self.var_vects = var_vects
        self.var_vers = var_vers

    def count_antecedents_len(self, df):
        """
        Fonction retournant un dictionnaire qui associe à chaque entier de 2 à n_antecedant_vers, le nombre
        d'observations labélisées par 1 qui contiennent ce nombre d'antécédants non nuls
        :param df: DataFrame de type vers

        :return: dict
        """
        ids = df.loc[:, self.var_id].to_list()
        count_ids = Counter(ids)
        dico_count = dict()
        for n_ant in range(2, self.n_antecedant_vers + 1):
            dico_count[n_ant] = len([k for k, v in count_ids.items() if v >= n_ant])
        dico_count[self.n_antecedant_vers + 1] = sum([(v - self.n_antecedant_vers) for k, v in count_ids.items()
                                                      if v > self.n_antecedant_vers])
        return dico_count

    def get_index_fake(self, df, n_0_general=1, coef_calage=10, graine=None, shuffle=True):
        """
        Retourne une liste d'indices dans laquelle piocher pour créer les éléments labélisés par 0 dans l'apprentissage
        :param df: DataFrame de type vers
        :param n_0_general: nombre d'élements labelisés par 0 à infliger à chaque élements de longueur
        n_antecedant_vers + 1 (default 1)
        :param coef_calage: nombre d'élements labelisés par 0 pour les débuts de vers (default 10)
        :param graine: graine aléatoire pour le tirage du reste des indices (default None)
        :param shuffle: mélanger la liste à la fin (default True)

        :return: liste des indices à piocher dans le DataFrame
        """
        m = df.shape[0]
        dico_count = self.count_antecedents_len(df=df)
        nb_labels0 = dico_count[self.n_antecedant_vers + 1] * n_0_general
        for k, v in dico_count.items():
            if k <= self.n_antecedant_vers:
                nb_labels0 += v * n_0_general * coef_calage
        repetition_idx_size = nb_labels0 // m
        reste_idx_size = nb_labels0 % m
        interval = np.arange(m)
        if graine is not None:
            np.random.seed(graine)
        reste_idx = np.random.choice(interval, reste_idx_size, replace=False)
        idx = np.concatenate([np.repeat(interval, repetition_idx_size), reste_idx])
        if shuffle:
            np.random.shuffle(idx)
        idx = list(idx.tolist())
        return idx

    def df2list_idx_train(self, df, n_0_general=1, n_surroundings=1, size_surroundings=6, coef_calage=10,
                          graine=None, shuffle=True):
        """
        Retourne la liste des indices des vers à prendre pour construire les matrices pour l'entraînement.
        -1 est placé pour les éléments vides
        :param df: DataFrame de type vers
        :param n_0_general: nombre d'élements labelisés par 0 à infliger à chaque élements de longueur
        n_antecedant_vers + 1 (default 1)
        :param n_surroundings: nombre d'éléments à tirer de l'entourage (default 1)
        :param size_surroundings: taille de l'entourage (default 6)
        :param coef_calage: nombre d'élements labelisés par 0 pour les débuts de vers (default 10)
        :param graine: graine aléatoire pour le tirage du reste des indices (default None)
        :param shuffle: mélanger la liste à la fin (default True)

        :return: liste de listes des indices
        """
        fake_ids = self.get_index_fake(df, n_0_general, coef_calage, graine, shuffle)
        id_poems_all = list(df.loc[:, self.var_id].to_list())
        count_id_poems = Counter(id_poems_all)
        id_poems = set([k for k, v in count_id_poems.items() if v >= 2])  # on enleve les poemes d'un seul vers
        idx_train = list()
        for id_poem in id_poems:
            rows_poem = list(np.flatnonzero(df.loc[:, self.var_id] == id_poem).tolist())
            len_poem = len(rows_poem)
            surroundings_id = list()
            for _ in range(n_surroundings):
                idx_sur = fake_surroundings(len_poem=len_poem, size_surroundings=size_surroundings)
                surroundings_id.extend(idx_sur)
            for row in range(1, len_poem):
                antecedants = rows_poem[max(0, row - self.n_antecedant_vers):row]

                # on rajoute - pour les débuts de vers
                idx_train.append([-1] * max(self.n_antecedant_vers - row, 0) + antecedants + [rows_poem[row]])
                for i_surrounding in range(n_surroundings):
                    sur = surroundings_id[row + i_surrounding * len_poem]
                    idx_train.append([-1] * max(self.n_antecedant_vers - row, 0) + antecedants + [rows_poem[sur]])
                for _ in range(n_0_general):
                    if row < self.n_antecedant_vers:
                        for _ in range(coef_calage):
                            fake_idx = fake_ids.pop()
                            idx_train.append([-1] * (self.n_antecedant_vers - row) + antecedants + [fake_idx])
                    else:
                        fake_idx = fake_ids.pop()
                        idx_train.append(antecedants + [fake_idx])
        if shuffle:
            np.random.shuffle(idx_train)
        return idx_train

    def phonemes2one_hot(self, phonemes):
        """
        :param phonemes: liste de la forme [vers_precedant_k, ..., vers_precedant_1, vers_cible]
        :return: matrice one hot correspondant aux phonèmes de la liste
        """
        n_p = len(self.p2idx.keys())
        m = len(phonemes)
        x_tout = np.zeros((m, self.t_p + 1, n_p))
        for i, phoneme in enumerate(phonemes):
            phoneme_long = "{blanks}{phoneme}".format(blanks=self.blank * (self.t_p + 1 - len(phoneme)),
                                                      phoneme=phoneme)
            assert len(phoneme_long) == self.t_p + 1, "phoneme long n'est pas de la meme longueur que la matrix"
            for j, p in enumerate(phoneme_long):
                x_tout[i, j, self.p2idx[p]] = 1
        return x_tout

    def labelizer(self, indexes):
        """
        Labélise une suite d'indices
        :param indexes: liste d'une suite d'indices
        :return: label 0 ou 1
        """
        if len(indexes) != self.n_antecedant_vers + 1:
            raise ValueError("La liste n'est pas de la bonne longueur")
        if indexes[self.n_antecedant_vers - 1] + 1 == indexes[self.n_antecedant_vers]:
            return 1
        else:
            return 0

    def liste2matrixes(self, liste_idx, df, labeliser=False):
        """
        Retourne les matrices nécessaires pour l'entraînement à partir des listes d'indices
        :param liste_idx: liste des indices
        :param df: Data Frame de type vers
        :param labeliser: labélisation des données (default False)
        :return: matrice des phonèmes et celle des vecteurs
        """
        # pour les vers vides, le word embedding correspond au vecteur de ft.get_word_vector("/s"), ici [0] * dim_words
        dict_empty_element = {self.var_id: -1, self.var_phonemes: '', self.var_vects: np.array([0] * self.dim_words)}
        df_empty = pd.DataFrame(columns=df.columns)
        df_empty = df_empty.append(dict_empty_element, ignore_index=True)
        df_copy = df.copy()
        df_copy.loc[-1] = df_empty.loc[0]
        m = len(liste_idx)
        n_p = len(self.p2idx)
        mat_phon = np.zeros((m, self.n_antecedant_vers + 1, self.t_p + 1, n_p))
        mat_vect = np.zeros((m, self.n_antecedant_vers + 1, self.dim_words))
        labels = list()
        for i, elements_idx in tqdm(enumerate(liste_idx)):
            mat_phon[i, :, :, :] = self.phonemes2one_hot(df_copy.loc[elements_idx, self.var_phonemes])
            mat_vect[i, :, :] = np.stack(df_copy.loc[elements_idx, self.var_vects])
            if labeliser:
                labels.append(self.labelizer(elements_idx))
        if labeliser:
            labels = np.array(labels)
            return mat_phon, mat_vect, labels
        else:
            return mat_phon, mat_vect

    def model(self, n_rnnphonvers=300, n_rnn_phon15=420, n_dense_particulier=50, n_dense_particulier2=30, n_dense1=25):
        # phonemes
        n_p = len(self.p2idx)
        phonemes_in = Input(shape=(self.n_antecedant_vers + 1, self.t_p + 1, n_p), name="phonemes_input")

        rnn_phon = GRU(units=n_rnnphonvers, activation="tanh", name="embedding_phonemes")

        phonems_v15 = TimeDistributed(rnn_phon, name="RNN_phonemes_vers_unique")(phonemes_in)
        phonems_v14 = Lambda(lambda x: x[:, :-1, :], output_shape=(8, n_rnnphonvers), name="phonemes_v")(phonems_v15)
        phonems_cible = Lambda(lambda x: x[:, -1, :], output_shape=(n_rnnphonvers,), name="phonemes_cible")(phonems_v15)

        rnn_phon_vers = LSTM(units=n_rnn_phon15, activation="tanh", name="RNN_phonemes_total")(phonems_v14)

        phonemes_tout = Concatenate(axis=1, name="concat_phonemes")([rnn_phon_vers, phonems_cible])
        phonemes_tout = Dense(units=n_dense_particulier, name="dense_phonemes")(phonemes_tout)
        phonemes_tout = LeakyReLU(0.2, name="activation_dense_phonemes")(phonemes_tout)
        phonemes_tout = Dropout(rate=0.1, name="dropout_phonemes")(phonemes_tout)
        phonemes_tout = BatchNormalization(name="batchnorm_phonemes")(phonemes_tout)
        phonemes_tout2 = Dense(units=n_dense_particulier2, name="dense_phonemes2")(phonemes_tout)
        phonemes_tout2 = LeakyReLU(0.2, name="activation_dense_phonemes2")(phonemes_tout2)
        phonemes_tout2 = Dropout(rate=0.1, name="dropout_phonemes2")(phonemes_tout2)
        phonemes_tout2 = BatchNormalization(name="batchnorm_phonemes2")(phonemes_tout2)

        # word embedding
        words_in = Input(shape=(self.n_antecedant_vers + 1, self.dim_words), name="words_input")

        emb_reduc = Sequential(name="dim_reduction")
        emb_reduc.add(Dense(units=n_rnnphonvers, activation="tanh"))
        emb_reduc.add(Lambda(lambda t: K.l2_normalize(1000*t, axis=1)))
        emb_reduc.add(Dropout(rate=0.1))
        emb_reduc.add(BatchNormalization())

        words_v15 = TimeDistributed(emb_reduc)(words_in)
        words_v14 = Lambda(lambda x: x[:, :-1, :], output_shape=(8, n_rnnphonvers), name="words_v")(words_v15)
        wordscible = Lambda(lambda x: x[:, -1, :], output_shape=(n_rnnphonvers,), name="wordscible")(words_v15)

        rnn_words = GRU(units=n_rnn_phon15, activation="tanh", name="RNN_words")(words_v14)

        words_tout = Concatenate(axis=1, name="concat_words")([rnn_words, wordscible])
        words_tout = Dense(units=n_dense_particulier, name="dense_words")(words_tout)
        words_tout = LeakyReLU(0.2, name="activation_dense_words")(words_tout)
        words_tout = Dropout(rate=0.1, name="dropout_words")(words_tout)
        words_tout = BatchNormalization(name="batchnorm_words")(words_tout)
        words_tout2 = Dense(units=n_dense_particulier2, name="dense_words2")(words_tout)
        words_tout2 = LeakyReLU(0.2, name="activation_dense_words2")(words_tout2)
        words_tout2 = Dropout(rate=0.1, name="dropout_words2")(words_tout2)
        words_tout2 = BatchNormalization(name="batchnorm_words2")(words_tout2)

        # rassemblement
        tout = Concatenate(axis=1, name="concat_tout")((phonemes_tout2, words_tout2))
        dense1 = Dense(units=n_dense1, name="dense_tout1")(tout)
        dense1 = LeakyReLU(0.2, name="activation_dense_tout1")(dense1)
        dense1 = Dropout(rate=0.1, name="dropout_tout1")(dense1)
        dense1 = BatchNormalization(name="batchnorm_tout1")(dense1)
        proba = Dense(units=1, activation="sigmoid", name="probability")(dense1)

        net = Model(inputs=[phonemes_in, words_in], outputs=proba)
        net.summary()
        return net

    def compile_train(self, mat_p, mat_v, labels, opt, epochs=10, batch_size=64):
        if self.net is None:
            self.net = self.model()
            self.net.summary()
        self.net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        self.net.fit([mat_p, mat_v], labels, epochs=epochs, batch_size=batch_size)
        return self.net

    def epoch_schlag(self, liste_idx, df, opt, split=10, batch_size=64, accelerator=None):
        m = len(liste_idx)
        width_split = m // split
        self.net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        for i in range(split):
            if i < split - 1:
                liste_idx_split = liste_idx[(i * width_split):((i + 1) * width_split)]
            else:
                liste_idx_split = liste_idx[(i * width_split):]
            mat_phon, mat_vect, labels = self.liste2matrixes(liste_idx=liste_idx_split, df=df, labeliser=True)
            if accelerator is not None:
                with accelerator:
                    self.net.fit([mat_phon, mat_vect], labels, epochs=1, batch_size=batch_size)
            else:
                self.net.fit([mat_phon, mat_vect], labels, epochs=1, batch_size=batch_size)
            del mat_phon, mat_vect, labels
        return self.net

    def save_net(self, path):
        self.net.save(path)

    def evaluate(self, df, n_0_general=1, coef_calage=10, n_surroundings=1, size_surroundings=6,
                 batch_size=128, graine=23):
        liste_idx = self.df2list_idx_train(df, n_0_general=n_0_general, coef_calage=coef_calage,
                                           n_surroundings=n_surroundings, size_surroundings=size_surroundings,
                                           graine=graine)
        mat_phon, mat_vect, labels = self.liste2matrixes(liste_idx=liste_idx, df=df, labeliser=True)
        results = self.net.evaluate([mat_phon, mat_vect], labels, batch_size=batch_size)
        return results

    def vers2matrixes(self, liste_vers, lecteur, ft, len_output=None):
        """
        Transforme une liste de vers en ses matrices correspondantes
        :param liste_vers:  liste de strings de vers
        :param lecteur: instance de Lecteur permettant la lecture des vers
        :param ft: modèle FastText dans lequel se projète les vecteurs de mots
        :param len_output: dimension 0 de la matrice (default n_antecedant_vers + 1)
        :return: tuple (matrice de phonemes, matrice de vecteurs)
        """
        if len_output is None:
            len_output = self.n_antecedant_vers + 1
        len_liste = len(liste_vers)
        phonemes = list()
        vects = list()
        if len_liste > len_output:
            raise ValueError
        elif len_liste < len_output:
            empty_p = ''
            empty_v = ft.get_sentence_vector("/s")
            n_empty = len_output - len_liste
            phonemes = [empty_p] * n_empty
            vects = [empty_v] * n_empty
        for vers in liste_vers:
            phoneme, vect = vers2phon_vect(vers, lecteur, ft)
            phonemes.append(phoneme)
            vects.append(vect)
        mat_phon = self.phonemes2one_hot(phonemes)
        mat_vect = np.stack(vects)
        return mat_phon, mat_vect

    def probas_candidats(self, df, liste_vers=None, lecteur=None, ft=None, mphon_prec=None, mvect_prec=None,
                         batch_size=128, split=5, accelerator=None):
        if (mphon_prec is None) or (mvect_prec is None):
            mphon_prec, mvect_prec = self.vers2matrixes(liste_vers, lecteur, ft,
                                                        len_output=self.n_antecedant_vers)
        m = df.shape[0]
        split_width = m // split
        pred = None
        for i in range(split):
            if i < split - 1:
                mphon_cibles = self.phonemes2one_hot(df.loc[(i*split_width):((i+1)*split_width - 1), self.var_phonemes])
                mvect_cibles = np.stack(df.loc[(i*split_width):((i+1)*split_width - 1), self.var_vects])
            else:
                mphon_cibles = self.phonemes2one_hot(df.loc[(i * split_width):, self.var_phonemes])
                mvect_cibles = np.stack(df.loc[(i * split_width):, self.var_vects])
                split_width = df.loc[(i * split_width):, self.var_phonemes].shape[0]
            mphon_prec_sized = np.repeat(mphon_prec[np.newaxis, :, :, :], split_width, axis=0)
            mvect_prec_sized = np.repeat(mvect_prec[np.newaxis, :, :], split_width, axis=0)
            mphon = np.concatenate([mphon_prec_sized, mphon_cibles[:, np.newaxis, :, :]], axis=1)
            mvect = np.concatenate([mvect_prec_sized, mvect_cibles[:, np.newaxis, :]], axis=1)
            if accelerator is None:
                prednouv = self.net.predict([mphon, mvect], batch_size=batch_size)
            else:
                with accelerator:
                    prednouv = self.net.predict([mphon, mvect], batch_size=batch_size)
            if i == 0:
                pred = prednouv
            else:
                pred = np.concatenate([pred, prednouv])
            del mphon_cibles, mvect_cibles, mphon_prec_sized, mvect_prec_sized, mphon, mvect
        assert pred.shape[0] == df.shape[0], "Le nombre de prédictions ne correspond pas à la taille du DataFrame"
        return pred

    def find_best_candidat(self, liste_vers=None, df=None, lecteur=None, ft=None, mphon_prec=None, mvect_prec=None,
                           batch_size=128, split=5, accelerator=None):
        pred = self.probas_candidats(liste_vers=liste_vers, df=df, lecteur=lecteur, ft=ft, mphon_prec=mphon_prec,
                                     mvect_prec=mvect_prec, batch_size=batch_size, split=split, accelerator=accelerator)
        candidat = np.argmax(pred)
        return df.iloc[candidat, :]

    def beam_search_write(self, liste_vers, df, vers_suivants=4, k=3, split=5, batch_size=128, accelerator=None,
                          **kwargs):
        lecteur = kwargs.get("lecteur", None)
        ft = kwargs.get("ft", None)
        mphon_prec = kwargs.get("mphon_prec", None)
        mvect_prec = kwargs.get("mvect_prec", None)

        if (mphon_prec is None) or (mvect_prec is None):
            mphon_prec, mvect_prec = self.vers2matrixes(liste_vers, lecteur, ft, len_output=self.n_antecedant_vers)

        kbest = [(liste_vers, "a", 0.0)]
        for n_vers_suivant in range(vers_suivants):
            tous_candidats = list()
            if n_vers_suivant > 0:
                mphon_prec = mphon_prec[1:, :, :]
                mvect_prec = mvect_prec[1:, :]
            for poem, index_last, score in kbest:
                if n_vers_suivant > 0:
                    phonemes_dernier = df.loc[index_last, self.var_phonemes]
                    phonemes_oh_dernier = self.phonemes2one_hot([phonemes_dernier])
                    mphon_prec = np.concatenate([mphon_prec[:(self.n_antecedant_vers - 1), :, :],
                                                 phonemes_oh_dernier], axis=0)

                    vect_dernier = df.loc[index_last, self.var_vects][np.newaxis, :]
                    mvect_prec = np.concatenate([mvect_prec[:(self.n_antecedant_vers - 1), :],
                                                 vect_dernier], axis=0)
                probas = self.probas_candidats(df=df, mphon_prec=mphon_prec, mvect_prec=mvect_prec,
                                               batch_size=batch_size, split=split, accelerator=accelerator)
                n_candidats = len(probas)
                for j in range(n_candidats):
                    vers_candidat = df.loc[j, self.var_vers]
                    proba_candidat = probas[j]
                    candidat = (poem + [vers_candidat], j, score - np.log(proba_candidat))
                    tous_candidats.append(candidat)

            ordered = sorted(tous_candidats, key=lambda scr: scr[2])
            kbest = ordered[:k]
            if n_vers_suivant == 0:
                print("Nouveau vers")
            else:
                print("{} nouveaux vers".format(n_vers_suivant + 1))
            if k >= 3:
                print("Harry :")
                print("\n".join([vers for vers in kbest[2][0]]))
                print("\n")
            if k >= 2:
                print("Dauphin :")
                print("\n".join([vers for vers in kbest[1][0]]))
                print("\n")
                print("Élu :")
            print("\n".join([vers for vers in kbest[0][0]]))
            print("\n")
            print("\n")
        return kbest

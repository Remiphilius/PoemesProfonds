from trad_chiffre_mot import tradn
import os
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import RegexpTokenizer
from keras import Input
from keras.layers import Bidirectional, LSTM, Dropout, RepeatVector, Concatenate, Dense, Activation, Dot, GRU
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


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
    x = Input(shape=(tx, n_l))
    c0 = Input(shape=(n_h1,), name='c0')
    h0 = Input(shape=(n_h1,), name='h0')
    c = c0
    h = h0
    outputs = list()  # initialisation de la derniere couche

    # c'est parti
    a = Bidirectional(LSTM(units=n_brnn1, return_sequences=True, name="LSTM_mot"))(x)
    a = Dropout(0.2, name="dropout_LSTM_orthographe")(a)
    for t in range(ty):
        # Attention
        h_rep = RepeatVector(tx, name="att_repeat_phoneme{}".format(t))(h)
        ah = Concatenate(axis=-1, name="att_concat_phoneme{}".format(t))([h_rep, a])
        energies = Dense(units=n_h1, activation="tanh", name="att_caractere_phoneme{}".format(t))(ah)
        energies = Dense(units=1, activation="relu", name="att_moyenne_phoneme{}".format(t))(energies)
        alpha = Activation("softmax", name="att_alpha_phoneme{}".format(t))(energies)
        context = Dot(axes=1, name="att_application_phoneme{}".format(t))([alpha, a])

        h, c = GRU(units=n_h1, activation='tanh', recurrent_activation='tanh', return_state=True,
                   name="LSTM_phoneme{}".format(t))(inputs=context, initial_state=c)
        h = Dropout(rate=0.1, name="dropout_phoneme{}".format(t))(h)
        c = Dropout(rate=0.1, name="dropout_memory_phoneme{}".format(t))(c)
        outy = Dense(activation="softmax", units=n_p, name="LSTM_{}".format(t))(h)
        outputs.append(outy)
    net = Model(inputs=[x, c0, h0], outputs=outputs)
    return net


def pos_tag(mots,
            jar=os.path.join(".", "models", "stanford-postagger", "stanford-postagger-3.8.0.jar"),
            mdl=os.path.join(".", "models", "stanford-postagger", "french-ud.tagger")):
    try:
        pos_tagger = StanfordPOSTagger(mdl, jar, encoding='utf8')
    except LookupError:
        java_path = r"C:\Program Files (x86)\Java\jre1.8.0_261\bin\java.exe"
        os.environ['JAVAHOME'] = java_path
        pos_tagger = StanfordPOSTagger(mdl, jar, encoding='utf8')
    tagged = pos_tagger.tag(mots)
    tags = [g for m, g in tagged]
    forced_det = ["au", "aux"]
    absent_of_table = ["PART", "SCONJ"]
    if any(item in mots for item in forced_det) or any(item in tags for item in absent_of_table):
        for i, couple in enumerate(tagged):
            mot = couple[0]
            gram = couple[1]
            if mot in forced_det:
                tagged[i] = (mot, "DET")
            if gram == "PART":
                tagged[i] = (mot, "ADV")
            if gram == "SCONJ":
                tagged[i] = (mot, "CONJ")
    return tagged


def check_liaison(ortho1, ortho2, phon1, phon2, nat1, nat2, phrase, **kwargs):
    """
    Fonction qui verifie si la liaison est possible entre deux mots
    :param ortho1: orthographe du mot en position 1
    :param ortho2: orthographe du mot en position 2
    :param phon1: phonemes du mot en position 1
    :param phon2: phonemes du mot en position 2
    :param nat1: nature du mot en position 1
    :param nat2: nature du mot en position 2
    :param phrase: phrase de contexte

    :return: booleen sur la possibilite de liaison
    """
    voyelles_p = kwargs.get("voyelles_p", ['a', 'E', '§', 'o', 'O', '1', 'i', '5', 'e', 'u', '@', '°', '9', 'y', '2'])
    y_p = kwargs.get("y_p", ['w', 'j', '8'])
    consonnes_liaisons = {'d': ['d'], 'p': ["p"], 'r': ["R"], 's': ['s', 'z'], 't': ['t'], 'x': ['s', 'z'],
                          'n': ['n', 'G'], 'z': ['z', 's']}
    liables = False
    mot2_voyelle = ((phon2[0] in y_p) or (phon2[0] in voyelles_p)) and (ortho2[0] != 'h')
    if mot2_voyelle:
        mot1_consonne_liaison = (ortho1[-1] in consonnes_liaisons.keys()) and\
                                (phon1[-1] not in consonnes_liaisons[ortho1[-1]])
        if mot1_consonne_liaison:
            mot1_dern_son_voyelle = (ortho1[-1] in consonnes_liaisons.keys()) and (phon1[-1] in voyelles_p)
            pas_ponctuation = (" ".join([ortho1, ortho2]) in phrase) or ("-".join([ortho1, ortho2]) in phrase)
            if pas_ponctuation:
                if (nat1 in ["NUM", "DET", "ADJ"]) and (nat2 in ["NOUN", "PROPN"]):
                    liables = True
                elif ortho1 in ["on", "nous", "vous", "ils", "elles", "en", "tout"] and nat2 in ["AUX", "VERB"]:
                    liables = True
                elif nat1 in ["AUX", "VERB"] and mot1_dern_son_voyelle:
                    liables = True
                elif nat1 in ["ADP"]:
                    liables = True
                elif (nat1 in ["NOUN"]) and (ortho1[-1] in ['s']) and (nat2 in ["ADJ"]):
                    liables = True
                elif (nat1 == "ADV") and (nat2 in ["ADV", "ADJ", "NOUN"]):
                    liables = True
                elif (ortho1 == "quand") and (nat2 not in ["AUX", "VERB"]):
                    liables = True
                elif (ortho1 == "plus") and (ortho2 == "ou"):
                    liables = True
                elif (ortho1 == "tout") and (ortho2 in ["à", "autour"]):
                    liables = True
    return liables


def liaison(ortho1, ortho2, phon1, phon2, nat1, nat2, phrase, **kwargs):
    dico_liaisons_simples = kwargs.get("dico_liaisons", {'d': 't', 'p': 'p', 's': 'z', 't': 't', 'x': 'z', 'z': 'z'})
    mots_nasale_simples = kwargs.get("mots_nasale_simples", ["aucun", "bien", "en", "on", "rien", "un", "non", "mon",
                                                             "ton", "son"])
    liaison_a_faire = check_liaison(ortho1, ortho2, phon1, phon2, nat1, nat2, phrase, **kwargs)
    if liaison_a_faire:
        derniere_lettre = ortho1[-1]
        if derniere_lettre in dico_liaisons_simples.keys():
            phon1 = "{}{}".format(phon1, dico_liaisons_simples[derniere_lettre])
        if derniere_lettre == 'r' and phon1[-1] == "e":  # comme "premier"
            phon1 = "{}{}".format(phon1[:-1], "ER")
        if derniere_lettre == 'n':  # comme "bon", "certain", "commun"
            dernier_phoneme = phon1[-1]
            if (ortho1 in mots_nasale_simples) or (dernier_phoneme == "1"):
                phon1 = "{}{}".format(phon1, 'n')
            else:
                if dernier_phoneme == "§":
                    phon1 = "{}{}".format(phon1[:-1], "On")
                if dernier_phoneme == "@" and ortho1[-2:] == "an":
                    phon1 = "{}{}".format(phon1[:-1], "an")
                if dernier_phoneme == "5":
                    if ortho1[-2:] == "en":
                        phon1 = "{}{}".format(phon1[:-1], "En")
                    if ortho1[-2:] == "in":
                        phon1 = "{}{}".format(phon1[:-1], "in")
                    if ortho1[-3:] in ["ein", "ain"]:
                        phon1 = "{}{}".format(phon1[:-2], "En")
    return phon1


def e_final(ortho1, ortho2, phon1, phon2, nat1, nat2, phrase, **kwargs):
    e_potentiel = (ortho1[-1] == 'e') or (ortho1[-2:] == 'es') or (ortho1[-3:] == 'ent')
    son_final = phon1[-1]
    son_initial = phon2[0]
    lettre_initiale = ortho2[0]
    consonnes_p = ['k', 'p', 'l', 't', 'R', 'j', 'f', 's', 'd', 'Z', 'n', 'b', 'v', 'g',
                   'v', 'g', 'm', 'z', 'w', 'S', 'N', '8', 'G', 'x']
    lien_mots = (son_final in consonnes_p) and ((son_initial in consonnes_p) or lettre_initiale == 'h')
    if e_potentiel and lien_mots:
        phon1 = "{}°".format(phon1)
    elif e_potentiel and (son_final in consonnes_p):  # "e" et liaison quand le 2e mot commence par une voyelle
        phon1_e = "{}°".format(phon1)
        phon1_e_liaison = liaison(ortho1, ortho2, phon1_e, phon2, nat1, nat2, phrase, **kwargs)
        if phon1_e != phon1_e_liaison:
            phon1 = phon1_e_liaison
    return phon1


def liaisons_tokens(mots, prononciation, pos_mots, phrase):
    n = len(prononciation)
    for i in range(n - 1):
        prononciation[i] = liaison(mots[i], mots[i + 1], prononciation[i], prononciation[i + 1],
                                   pos_mots[i][1], pos_mots[i + 1][1], phrase.lower())
    return prononciation


def e_final_tokens(mots, prononciation, pos_mots, phrase):
    n = len(prononciation)
    for i in range(n - 1):
        prononciation[i] = e_final(mots[i], mots[i + 1], prononciation[i], prononciation[i + 1],
                                   pos_mots[i][1], pos_mots[i + 1][1], phrase)
    return prononciation


class Lecteur:
    """"Classe definissant le lecteur
    """

    def __init__(self, tx, ty, l2idx, p2idx, dico_unique, dico_multiple, n_brnn1=90, n_h1=80, net=None, blank="_"):
        self.tx = tx
        self.ty = ty
        self.l2idx = l2idx
        self.p2idx = p2idx
        self._dico_unique = dico_unique
        self._ortho_unique = dico_unique.keys()
        self._dico_multiple = dico_multiple
        self._ortho_multiple = list(set([w for w, _ in dico_multiple.keys()]))
        self.n_brnn1 = n_brnn1
        self.n_h1 = n_h1
        self.net = net
        self.blank = blank
        self.count_lecture = 0

    # setters et getters
    def _get_dico_unique(self):
        return self._dico_unique

    def _set_dico_unique(self, dico_unique):
        self._dico_unique = dico_unique
        self._ortho_unique = dico_unique.keys()

    def _get_ortho_unique(self):
        return self._ortho_unique

    def _set_ortho_unique(self, valeur):
        raise AttributeError("ortho_unique ne peut pas etre modifie")

    def _get_dico_multiple(self):
        return self._dico_multiple

    def _set_dico_multiple(self, dico_multiple):
        self._dico_multiple = dico_multiple
        self._ortho_multiple = list(set([w for w, _ in dico_multiple.keys()]))

    def _get_ortho_multiple(self):
        return self._ortho_multiple

    def _set_ortho_multiple(self, valeur):
        raise AttributeError("ortho_multiple ne peut pas etre modifie")

    dico_unique = property(fget=_get_dico_unique, fset=_set_dico_unique)
    ortho_unique = property(fget=_get_ortho_unique, fset=_set_ortho_unique)
    dico_multiple = property(fget=_get_dico_multiple, fset=_set_dico_multiple)
    ortho_multiple = property(fget=_get_ortho_multiple, fset=_set_ortho_multiple)

    # methodes
    def one_hot_from_list(self, data):
        """
        :param data: liste des couples (mot, phonemes)

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
        x = Input(shape=(self.tx, n_l), name="mot")
        c0 = Input(shape=(self.n_h1,), name='c0')
        c = c0
        h0 = Input(shape=(self.n_h1,), name='h0')
        h = h0
        outputs = list()  # initialisation de la derniere couche

        # c'est parti
        a = Bidirectional(LSTM(units=self.n_brnn1, return_sequences=True, name="LSTM_orthographe"))(x)
        a = Dropout(0.2, name="dropout_LSTM_orthographe")(a)
        for t in range(self.ty):
            # Attention
            h_rep = RepeatVector(self.tx, name="att_repeat_phoneme{}".format(t))(h)
            ah = Concatenate(axis=-1, name="att_concat_phoneme{}".format(t))([h_rep, a])
            energies = Dense(units=self.n_h1, activation="tanh", name="att_caractere_phoneme{}".format(t))(ah)
            energies = Dense(units=1, activation="relu", name="att_moyenne_phoneme{}".format(t))(energies)
            alpha = Activation("softmax", name="att_alpha_phoneme{}".format(t))(energies)
            context = Dot(axes=1, name="att_application_phoneme{}".format(t))([alpha, a])

            h, c = GRU(units=self.n_h1, activation='tanh', recurrent_activation='sigmoid', return_state=True,
                       name="GRU_phoneme{}".format(t))(inputs=context, initial_state=c)

            h = Dropout(rate=0.1, name="dropout_phoneme{}".format(t))(h)
            c = Dropout(rate=0.1, name="dropout_memory_phoneme{}".format(t))(c)

            outy = Dense(activation="softmax", units=n_p, name="softmax_phoneme_{}".format(t))(h)
            outputs.append(outy)
        net = Model(inputs=[x, c0, h0], outputs=outputs)
        return net

    def compile_train(self, x, y, epochs=10, batch_size=64, opt=Adam()):
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

    def regex_lecteur(self, phrase, trad_numbers=False):
        if trad_numbers:
            reg0 = "0-9"
        else:
            reg0 = ""
        skip = [" ", ".", '-', self.blank]  # elements a ne pas prendre en compte lors de l'expression reguliere
        trait_dunion = False
        for c in self.l2idx.keys():
            if c == '-':
                trait_dunion = True
            if c not in skip:
                reg0 = "{}{}".format(reg0, c)
        if trait_dunion:
            reg0 = "{}-".format(reg0)
        reg = "[{}]+".format(reg0)
        res = RegexpTokenizer(reg).tokenize(phrase)
        return res

    def tirets_appostrophes(self, a_decouper):
        # appostrophes
        if a_decouper[0] == "'":
            a_decouper = a_decouper[1:]
        a_decouper = [splitted for splitted in a_decouper.split("'") if len(splitted) > 0]
        if len(a_decouper) > 1:
            for i in range(len(a_decouper) - 1):
                a_decouper[i] = "{}'".format(a_decouper[i])

        # traits-d'union
        decoupe = list()
        if a_decouper[-1][-1] == '-':
            a_decouper[-1] = a_decouper[-1][:-1]
        for partie in a_decouper:
            partie_splitted = [par for par in partie.split("-") if len(par) > 0]
            if len(partie_splitted) > 1:
                for i in range(1, len(partie_splitted)):
                    partie_splitted[i] = "-{}".format(partie_splitted[i])
            decoupe.extend(partie_splitted)

        # recherche dans le vocabulaire
        tokens = list()  # liste de tokens a retourner
        while len(decoupe) > 0:  # decoupe est reduite a mesure que tokens se remplit
            p = len(decoupe)
            if decoupe[0][0] == '-':  # si le premier caractere est un trait d'union, on l'enleve
                decoupe[0] = decoupe[0][1:]
            element = "".join(decoupe)  # on forme un mot avec les elements
            element_know = (element in self.ortho_unique) or (element in self.ortho_multiple)
            while not element_know and p > 1:  # le nombre d'elements diminue jusqu'a ce qu'un mot connu apparaisse
                p -= 1
                element = "".join(decoupe[:p])
                element_know = (element in self.ortho_unique) or (element in self.ortho_multiple)
            tokens.append(element)
            decoupe = decoupe[p:]  # on eleve de decoupe les parties tokenisees
        return tokens

    def tokenizer(self, phrase):
        """
        Tokeniseur maison qui sépare autour des espaces et les appostophes des mots plus courts que deux chars

        :param phrase
        :return: phrase tokenisee
        """
        # minuscules
        phrase = phrase.lower()

        # oe et ae
        phrase = phrase.replace("œ", "oe")
        phrase = phrase.replace("æ", "ae")

        # appostrophes
        phrase = phrase.replace("’", "'")

        # regex
        tokens = self.regex_lecteur(phrase, trad_numbers=True)

        # appostrophes/traits-d'unions
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if any(appostrophe_ou_trait in tok for appostrophe_ou_trait in ['-', "'"]) and len(tok) > 1:
                tokens.pop(i)
                tokens_to_add = self.tirets_appostrophes(tok)
                for token_to_add in tokens_to_add:
                    tokens.insert(i, token_to_add)
                    i += 1
            else:
                i += 1

        # nombres
        for i, tok in enumerate(tokens):
            if tok.isdigit():
                tokens[i] = tradn(int(tok))
        phrase = " ".join(tokens)
        tokens = self.regex_lecteur(phrase, trad_numbers=False)
        return tokens

    def tokenizer_poem(self, poem):
        print(self.count_lecture)
        self.count_lecture += 1
        poem_tokens = list()
        for strophe in poem:
            strophe_tokens = list()
            for ver in strophe:
                ver_tokens = self.tokenizer(ver)
                strophe_tokens.append(ver_tokens)
            poem_tokens.append(strophe_tokens)
        return poem_tokens

    def lire_nn(self, mots):
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

        # creation dico
        prononciation = dict()
        for i, mot in enumerate(mots):
            prononciation_mot = ""
            for idx in pron_int[i, :].tolist():
                prononciation_mot = "{phons}{p}".format(phons=prononciation_mot,
                                                        p=idx2p[idx]).replace(self.blank, "")
            prononciation[mot] = prononciation_mot
        return prononciation

    def lire_mots(self, mots):
        # dicos
        mots_unique = set()
        mots_multiples = set()
        mots_nn = set()
        pos_mots = pos_tag(mots)
        dico_nn = 0
        for mot in mots:
            if mot in self.ortho_unique:
                mots_unique.update([mot])
            elif mot in self.ortho_multiple:
                mots_multiples.update([mot])
            else:
                mots_nn.update([mot])
        if len(mots_multiples) > 0:
            for m, p in pos_mots:
                if m in mots_multiples and (m, p) not in self.dico_multiple.keys():
                    mots_multiples.remove(m)
                    mots_nn.update([m])
        if len(mots_nn) > 0:
            dico_nn = self.lire_nn(mots_nn)
        prononciation = list()
        for i, mot in enumerate(mots):
            prononciation_mot = ""
            if mot in mots_unique:
                prononciation_mot = self.dico_unique[mot]
            elif mot in mots_multiples:
                prononciation_mot = self.dico_multiple[pos_mots[i]]
            elif mot in mots_nn:
                prononciation_mot = dico_nn[mot]
            prononciation.append(prononciation_mot)
        return prononciation

    def lire_vers(self, vers, count=False):
        """
        Lit un vers
        :param vers: Chaine de caracteres du vers
        :param count: Si l'on souhaite compter le nombre de vers que l'on lit
        :return: vers lu sous la forme d'une string de phonemes non espaces
        """
        if count:
            print(self.count_lecture)
            self.count_lecture += 1
        tokens = self.tokenizer(vers)
        pos = pos_tag(tokens)
        mots_lus = self.lire_mots(tokens)
        mots_lus_avec_e = e_final_tokens(tokens, mots_lus, pos, vers)
        mots_lus_liaisons = liaisons_tokens(tokens, mots_lus_avec_e, pos, vers)
        vers_lu = "".join(mots_lus_liaisons)
        return vers_lu

    def lire_strophe(self, strophe, ponctuation=None):
        if ponctuation is None:
            ponctuation = ['.', ',', '!', '?', "…"]

        # separation en phrases
        text_strophe = " ".join(strophe)
        reg0 = "".join(ponctuation)
        reg0 = "(?:(?![{caracteres_a_eviter}]).)+".format(caracteres_a_eviter=reg0)
        phrases_strophe = [phrase for phrase in RegexpTokenizer(reg0).tokenize(text_strophe) if len(phrase) > 0]

        # lecture des phrases
        phrases_lues = list()
        pos_phrases = list()
        for phrase in phrases_strophe:
            tokens_phrase = self.tokenizer(phrase)
            phrases_lues.extend(self.lire_mots(tokens_phrase))
            pos_phrases.extend(pos_tag(tokens_phrase))
        return phrases_lues, pos_phrases

    def lire_poem(self, poem_tokens, poem_text):
        print(self.count_lecture)
        self.count_lecture += 1
        poem_phonemes_tokens = list()
        assert len(poem_tokens) == len(poem_text), "Les listes de textes et des tokens ont des longueurs differentes"
        for idx_s, strophe in enumerate(poem_tokens):
            strophe_text = poem_text[idx_s]
            strophe_phonemes_tokens = list()
            pos_strophe = list()
            phrase_a_lire = list()
            for ver_tokens in strophe:
                n_tok_ver = len(ver_tokens)
                phrase_a_lire.extend(ver_tokens)
                strophe_phonemes_tokens.append(n_tok_ver * [''])
                pos_strophe.append(n_tok_ver * [''])
            phrase_lue = self.lire_mots(phrase_a_lire)
            pos_phrase = pos_tag(phrase_a_lire)
            i = 0
            j = 0
            for idx, phoneme_mot in enumerate(phrase_lue):
                while len(strophe_phonemes_tokens[j]) == 0:
                    strophe_phonemes_tokens[j] = ""
                    pos_strophe[j] = []
                    i = 0
                    j += 1
                strophe_phonemes_tokens[j][i] = phoneme_mot
                pos_strophe[j][i] = pos_phrase[idx]
                i += 1
                if i == len(strophe_phonemes_tokens[j]):
                    i = 0
                    j += 1
            for idx in range(len(strophe_phonemes_tokens)):
                if len(strophe_phonemes_tokens[idx]) > 1:
                    strophe_phonemes_tokens[idx] = e_final_tokens(strophe[idx], strophe_phonemes_tokens[idx],
                                                                  pos_strophe[idx], strophe_text[idx])
                    strophe_phonemes_tokens[idx] = liaisons_tokens(strophe[idx],
                                                                   strophe_phonemes_tokens[idx], pos_strophe[idx],
                                                                   strophe_text[idx])
                    strophe_phonemes_tokens[idx] = "".join(strophe_phonemes_tokens[idx])
            if strophe_phonemes_tokens == [[]]:
                strophe_phonemes_tokens = ['']
            poem_phonemes_tokens.append(strophe_phonemes_tokens)
        return poem_phonemes_tokens

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


if __name__ == "lecture":
    os.environ['JAVAHOME'] = r"C:\Program Files (x86)\Java\jre1.8.0_261\bin\java.exe"

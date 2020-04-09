import pandas as pd


def import_lexique_as_df(path=r".\lexique383.xlsx"):
    """importe le lexique

    :argument
    path=r".\lexique383.xlsx" chemin du fichier

    :return pd.dataframe
    """
    df = pd.read_excel(path)
    df.iloc[:, 0] = df.iloc[:, 0].fillna(value="nan")  # transforme NaN en "nan"
    df.iloc[:, 1] = pd.DataFrame(df.iloc[:, 1]).applymap(str)  # convertit phonemes en str
    return df


def accent_e_fin(df, motsvar="1_ortho", phonvar="2_phon", **kwargs):
    """"
    :argument
    df: pd.dataframe contenant le lexique
    motsvar="1_ortho" variable de df contenant les orthographes
    phonvar="2_phon" variable de df contenant les phonemes auxquels il faut ajouter le E final
    phoneme="°" phoneme du E final
    pcvvar="18_p_cvcv" variable du df contenant les voyelles et consonnes phonemes

    :return
    pd.dataframe avec phoneme a la fin de phonvar pour signifier les E
    """
    phoneme = kwargs.get("phoneme", "°")
    pcvvar = kwargs.get("pcvvar", "18_p_cvcv")

    # recuperation des mots avec un E final et un phoneme final qui n'est pas une voyelle
    e_ended = df[motsvar].apply(lambda x: (x[-1] == "e"))
    mute_e = df[pcvvar].apply(lambda x: (x[-1] in ["C", "Y"]))
    idx = e_ended & mute_e

    # ajout du E prononce
    df.loc[idx, phonvar] = df.loc[idx, phonvar].apply(lambda x: "{origin}{E}".format(origin=x, E=phoneme))
    return df


def set_ortho2phon(df, mots="1_ortho", phon="2_phon", occurances="10_freqlivres", accent_e=False, **kwargs):
    """crée un dictionnaire mappant pour chaque mot à sa prononciation

    :argument
    df: pd.dataframe contenant le lexique

    :return dict, pd.DataFrame
    """
    # ajout de l'accent au e a la fin des mots
    if accent_e:
        df = accent_e_fin(df, motsvar=mots, phonvar=phon, **kwargs)
    # creation df rassemblant la frequence de la prononciation de chaque orthographe
    df_occ = df[[mots, phon, occurances]].groupby([mots, phon], as_index=False).agg({occurances: "sum"})

    # on ne garde que les phonemes qui apparaissent le plus par orthographe
    idx = df_occ.groupby([mots])[occurances].transform(max) == df_occ[occurances]
    df_o2p = df_occ[[mots, phon]][idx]

    dict_o2p = pd.Series(df_o2p.iloc[:, 1].values, index=df_o2p.iloc[:, 0]).to_dict()

    return dict_o2p, df_occ


def chars2idx(df, mots="1_ortho", phon="2_phon", blank="_"):
    """
    :param df: pd.dataframe contenant le lexique
    :param mots: "1_ortho" variable de df contenant les orthographes
    :param phon: "2_phon" variable de df contenant les phonemes

    :return: 2 dictionnaires caractere indices des lettres et des ohonemes
    """
    ltrs = list()
    phons = list()
    m = df.shape[0]
    tx = 0
    ty = 0
    for i in range(m):
        mot = str(df.loc[i, mots])
        if len(mot) > tx:
            tx = len(mot)
        for ltr in mot:
            if ltr not in ltrs:
                ltrs.append(ltr)
        prononciation = str(df.loc[i, phon])
        if len(prononciation) > ty:
            ty = len(prononciation)
        for ph in prononciation:
            if ph not in phons:
                phons.append(ph)
    ltr2idx = {blank: len(ltrs)}
    phon2idx = {blank: len(phons)}
    for i, v in enumerate(ltrs):
        ltr2idx[v] = i
    for i, v in enumerate(phons):
        phon2idx[v] = i
    return ltr2idx, phon2idx, tx, ty

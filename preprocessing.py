import pandas as pd
import time


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


    :return dict
    """
    # ajout de l'accent au e a la fin des mots
    if accent_e:
        df = accent_e_fin(df, motsvar=mots, phonvar=phon, **kwargs)
    # creation df rassemblant la frequence de la prononciation de chaque orthographe
    df_occ = df[[mots, phon, occurances]].groupby([mots, phon], as_index=False).agg({occurances: "sum"})

    # on ne garde que les phonemes qui apparaissent le plus par orthographe
    idx = df_occ.groupby([mots])[occurances].transform(max) == df_occ[occurances]
    df_uniqueorth = df_occ[[mots, phon]][idx]

    return pd.Series(df_uniqueorth.iloc[:, 1].values, index=df_uniqueorth.iloc[:, 0]).to_dict()


now = time.time()
data = import_lexique_as_df()
then = time.time()
print(data.shape)
print(then - now)

dico2p = set_ortho2phon(data, accent_e=True)

phonetique = list()
for k, v in dico2p.items():
    for c in v:
        if c not in phonetique:
            print("{} : {} : {}".format(c, k, v))
            phonetique.append(c)

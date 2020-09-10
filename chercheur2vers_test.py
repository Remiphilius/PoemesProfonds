import unittest
import fasttext.util
from keras.models import load_model
import preprocessing as pp
import lecture as lc
from chercheur2vers import *


class Chercheur2VersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # df_w2p = pd.read_pickle(r"C:\Users\remif\PycharmProjects\PoemesProfonds\data\df_w2p.pkl")
        # _, phon2idx, _, _ = pp.chars2idx(df_w2p)
        dico_u, dico_m, df_w2p = pd.read_pickle(r"C:\Users\remif\PycharmProjects\PoemesProfonds\data\dicos.pickle")
        ltr2idx, phon2idx, t_x, t_y = pp.chars2idx(df_w2p)
        cls.checheur = Chercheur2Vers(t_p=50, p2idx=phon2idx, n_antecedant_vers=4)
        cls.checheur8 = Chercheur2Vers(t_p=50, p2idx=phon2idx, n_antecedant_vers=8)
        # cls.ft = fasttext.load_model('cc.fr.300.bin')
        # model_lire = load_model(r"C:\Users\remif\PycharmProjects\PoemesProfonds\models\lecteur\CE1_T12_l10.h5")
        # cls.lecteur = lc.Lecteur(t_x, t_y, ltr2idx, phon2idx, dico_u, dico_m,
        #                          n_brnn1=90, n_h1=80, net=model_lire, blank="_")

    def test_count_antecedents_len(self):
        vers_test = pd.DataFrame({"id": [1, 1, 1, 1, 1, 1, 1, 1, 23, 23, 23, 555, 555, 555, 555, 9, 9, 666, 666, 666,
                                         666, 84, 84, 84, 84, 84, 84, 98, 98, 98, 98, 98, 7]})
        dict_attendu = dict({2: 7, 3: 6, 4: 5, 5: 7})
        self.assertEqual(self.checheur.count_antecedents_len(vers_test), dict_attendu)

    def test_get_index_fake(self):
        lll = [1] * 20 + [6] * 60 + [3] * 1 + [4] * 3 + [5] * 4 + [10] * 5 + [21] * 2
        df_test = pd.DataFrame({"id": lll})
        resultat_attendu = list(np.repeat(np.arange(95), 3).tolist()) + [29, 93, 46, 3, 50, 82, 61, 79, 17, 9, 78]
        self.assertListEqual(self.checheur.get_index_fake(df_test, n_0_general=2, coef_calage=5, graine=23,
                                                          shuffle=False), resultat_attendu)
        self.assertNotEqual(self.checheur.get_index_fake(df_test, n_0_general=2, coef_calage=5, graine=23,
                                                         shuffle=True), resultat_attendu)

    def test_df2list_idx_train(self):
        lll = [1] * 7 + [3] * 1 + [10] * 5 + [21] * 2
        res_attendu = [[-1, -1, -1, 0, 1], [-1, -1, -1, 0, 14], [-1, -1, -1, 0, 5], [-1, -1, -1, 0, 10],
                       [-1, -1, -1, 0, 0],
                       [-1, -1, 0, 1, 2], [-1, -1, 0, 1, 4], [-1, -1, 0, 1, 11], [-1, -1, 0, 1, 14],
                       [-1, -1, 0, 1, 14], [-1, 0, 1, 2, 3], [-1, 0, 1, 2, 13],
                       [-1, 0, 1, 2, 13], [-1, 0, 1, 2, 12], [-1, 0, 1, 2, 12],
                       [0, 1, 2, 3, 4], [0, 1, 2, 3, 11], [0, 1, 2, 3, 11],
                       [1, 2, 3, 4, 5], [1, 2, 3, 4, 10], [1, 2, 3, 4, 10],
                       [2, 3, 4, 5, 6], [2, 3, 4, 5, 9], [2, 3, 4, 5, 9],
                       [-1, -1, -1, 8, 9], [-1, -1, -1, 8, 8], [-1, -1, -1, 8, 8], [-1, -1, -1, 8, 7],
                       [-1, -1, -1, 8, 7],
                       [-1, -1, 8, 9, 10], [-1, -1, 8, 9, 6], [-1, -1, 8, 9, 6], [-1, -1, 8, 9, 5], [-1, -1, 8, 9, 5],
                       [-1, 8, 9, 10, 11], [-1, 8, 9, 10, 4], [-1, 8, 9, 10, 4], [-1, 8, 9, 10, 3], [-1, 8, 9, 10, 3],
                       [8, 9, 10, 11, 12], [8, 9, 10, 11, 2], [8, 9, 10, 11, 2],
                       [-1, -1, -1, 13, 14], [-1, -1, -1, 13, 1], [-1, -1, -1, 13, 1], [-1, -1, -1, 13, 0],
                       [-1, -1, -1, 13, 0]]
        self.assertListEqual(self.checheur.df2list_idx_train(pd.DataFrame({"id": lll}), graine=23, shuffle=False,
                                                             n_0_general=2, coef_calage=2, n_surroundings=0,
                                                             size_surroundings=5), res_attendu)
        lll8 = [1] * 4 + [3] * 1 + [10] * 10 + [21] * 2
        res_attendu8 = [[-1, -1, -1, -1, -1, -1, -1, 0, 1],
                        [-1, -1, -1, -1, -1, -1, -1, 0, 16],
                        [-1, -1, -1, -1, -1, -1, -1, 0, 15],
                        [-1, -1, -1, -1, -1, -1, -1, 0, 11],
                        [-1, -1, -1, -1, -1, -1, -1, 0, 7],
                        [-1, -1, -1, -1, -1, -1, 0, 1, 2],
                        [-1, -1, -1, -1, -1, -1, 0, 1, 14],
                        [-1, -1, -1, -1, -1, -1, 0, 1, 3],
                        [-1, -1, -1, -1, -1, -1, 0, 1, 13],
                        [-1, -1, -1, -1, -1, -1, 0, 1, 12],
                        [-1, -1, -1, -1, -1, 0, 1, 2, 3],
                        [-1, -1, -1, -1, -1, 0, 1, 2, 5],
                        [-1, -1, -1, -1, -1, 0, 1, 2, 4],
                        [-1, -1, -1, -1, -1, 0, 1, 2, 1],
                        [-1, -1, -1, -1, -1, 0, 1, 2, 2],
                        [-1, -1, -1, -1, -1, -1, -1, 5, 6],
                        [-1, -1, -1, -1, -1, -1, -1, 5, 10],
                        [-1, -1, -1, -1, -1, -1, -1, 5, 0],
                        [-1, -1, -1, -1, -1, -1, -1, 5, 16],
                        [-1, -1, -1, -1, -1, -1, -1, 5, 16],
                        [-1, -1, -1, -1, -1, -1, 5, 6, 7],
                        [-1, -1, -1, -1, -1, -1, 5, 6, 15],
                        [-1, -1, -1, -1, -1, -1, 5, 6, 15],
                        [-1, -1, -1, -1, -1, -1, 5, 6, 14],
                        [-1, -1, -1, -1, -1, -1, 5, 6, 14],
                        [-1, -1, -1, -1, -1, 5, 6, 7, 8],
                        [-1, -1, -1, -1, -1, 5, 6, 7, 13],
                        [-1, -1, -1, -1, -1, 5, 6, 7, 13],
                        [-1, -1, -1, -1, -1, 5, 6, 7, 12],
                        [-1, -1, -1, -1, -1, 5, 6, 7, 12],
                        [-1, -1, -1, -1, 5, 6, 7, 8, 9],
                        [-1, -1, -1, -1, 5, 6, 7, 8, 11],
                        [-1, -1, -1, -1, 5, 6, 7, 8, 11],
                        [-1, -1, -1, -1, 5, 6, 7, 8, 10],
                        [-1, -1, -1, -1, 5, 6, 7, 8, 10],
                        [-1, -1, -1, 5, 6, 7, 8, 9, 10],
                        [-1, -1, -1, 5, 6, 7, 8, 9, 9],
                        [-1, -1, -1, 5, 6, 7, 8, 9, 9],
                        [-1, -1, -1, 5, 6, 7, 8, 9, 8],
                        [-1, -1, -1, 5, 6, 7, 8, 9, 8],
                        [-1, -1, 5, 6, 7, 8, 9, 10, 11],
                        [-1, -1, 5, 6, 7, 8, 9, 10, 7],
                        [-1, -1, 5, 6, 7, 8, 9, 10, 7],
                        [-1, -1, 5, 6, 7, 8, 9, 10, 6],
                        [-1, -1, 5, 6, 7, 8, 9, 10, 6],
                        [-1, 5, 6, 7, 8, 9, 10, 11, 12],
                        [-1, 5, 6, 7, 8, 9, 10, 11, 5],
                        [-1, 5, 6, 7, 8, 9, 10, 11, 5],
                        [-1, 5, 6, 7, 8, 9, 10, 11, 4],
                        [-1, 5, 6, 7, 8, 9, 10, 11, 4],
                        [5, 6, 7, 8, 9, 10, 11, 12, 13],
                        [5, 6, 7, 8, 9, 10, 11, 12, 3],
                        [5, 6, 7, 8, 9, 10, 11, 12, 3],
                        [6, 7, 8, 9, 10, 11, 12, 13, 14],
                        [6, 7, 8, 9, 10, 11, 12, 13, 2],
                        [6, 7, 8, 9, 10, 11, 12, 13, 2],
                        [-1, -1, -1, -1, -1, -1, -1, 15, 16],
                        [-1, -1, -1, -1, -1, -1, -1, 15, 1],
                        [-1, -1, -1, -1, -1, -1, -1, 15, 1],
                        [-1, -1, -1, -1, -1, -1, -1, 15, 0],
                        [-1, -1, -1, -1, -1, -1, -1, 15, 0]]
        self.assertListEqual(self.checheur8.df2list_idx_train(pd.DataFrame({"id": lll8}), graine=23, shuffle=False,
                                                              n_0_general=2, coef_calage=2, n_surroundings=0,
                                                              size_surroundings=5), res_attendu8)

    def test_phonemes2one_hot(self):
        lll = pd.Series(['', "akpEl§tRjofOsidZn15ebuv@g°m9zwySN82GxakpEl§tRjofOs", '', "Remi"])
        empty_char = np.array([0] * len(self.checheur.p2idx))
        empty_char[self.checheur.p2idx[self.checheur.blank]] = 1
        empty_vers = np.repeat(empty_char[np.newaxis, :], self.checheur.t_p + 1, axis=0)
        nonblank_keys = list(self.checheur.p2idx.keys())[1:]
        n_keys = len(nonblank_keys)
        complet_vers = np.zeros((self.checheur.t_p + 1, len(self.checheur.p2idx)))
        complet_vers[0, self.checheur.p2idx[self.checheur.blank]] = 1
        for i in range(1, self.checheur.t_p + 1):
            complet_vers[i, self.checheur.p2idx[nonblank_keys[(i - 1) % n_keys]]] = 1
        remi_vers = np.zeros((self.checheur.t_p + 1, len(self.checheur.p2idx)))
        for i in range(self.checheur.t_p + 1 - 4):
            remi_vers[i, self.checheur.p2idx[self.checheur.blank]] = 1
        remi_vers[self.checheur.t_p + 1 - 4, self.checheur.p2idx['R']] = 1
        remi_vers[self.checheur.t_p + 1 - 3, self.checheur.p2idx['e']] = 1
        remi_vers[self.checheur.t_p + 1 - 2, self.checheur.p2idx['m']] = 1
        remi_vers[self.checheur.t_p + 1 - 1, self.checheur.p2idx['i']] = 1
        res_attendu = np.stack([empty_vers, complet_vers, empty_vers, remi_vers])
        np.testing.assert_array_equal(self.checheur.phonemes2one_hot(lll), res_attendu)

    def test_labelizer(self):
        positif = [-1, -1, -1, 23, 24]
        negatif = [956, 957, 958, 959, 665]
        erreur = [-1, -1, 2, 3, 4, 5]
        self.assertEqual(self.checheur.labelizer(positif), 1)
        self.assertEqual(self.checheur.labelizer(negatif), 0)
        self.assertRaises(ValueError, self.checheur.labelizer, erreur)

    @unittest.skip("Il est long sa mère et il marche")
    def test_liste2matrixes(self):
        ft = fasttext.load_model('cc.fr.300.bin')
        df_test = pd.DataFrame({"id": [2, 2, 2, 2, 2, 2, 3, 6, 6],
                                "phonemes": ["Sval", "8itR", "v2lsyRson", "potOt", "§bEz", "@s@", "wid", "x°", "Zjx"],
                                "vect": [ft.get_sentence_vector(phrase) for phrase in ["cheval", "huître",
                                                                                       "Veule sur Saône", "pas tate",
                                                                                       "on baize", "En sang",
                                                                                       "Mauvaises herbes", "Salamanque",
                                                                                       "J'aime la vie mais aussi P"]]})
        empty_vers_v = ft.get_word_vector("/s")
        p1 = self.checheur.phonemes2one_hot(['', '', '', "Sval", "8itR"])
        p2 = self.checheur.phonemes2one_hot(['', '', '', "Sval", "8itR"])
        p3 = self.checheur.phonemes2one_hot(['', '', "Sval", "8itR", "v2lsyRson"])
        p4 = self.checheur.phonemes2one_hot(['', '', "Sval", "8itR", "x°"])
        p5 = self.checheur.phonemes2one_hot(['', "Sval", "8itR", "v2lsyRson", "potOt"])
        p6 = self.checheur.phonemes2one_hot(['', "Sval", "8itR", "v2lsyRson", "Zjx"])
        p7 = self.checheur.phonemes2one_hot(["Sval", "8itR", "v2lsyRson", "potOt", "§bEz"])
        p8 = self.checheur.phonemes2one_hot(["Sval", "8itR", "v2lsyRson", "potOt", "@s@"])
        p9 = self.checheur.phonemes2one_hot(["8itR", "v2lsyRson", "potOt", "§bEz", "@s@"])
        p10 = self.checheur.phonemes2one_hot(["8itR", "v2lsyRson", "potOt", "§bEz", "§bEz"])
        p11 = self.checheur.phonemes2one_hot(['', '', '', "x°", "Zjx"])
        p12 = self.checheur.phonemes2one_hot(['', '', '', "x°", "v2lsyRson"])
        mat_p_attendue = np.stack([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12])

        v1 = np.stack([empty_vers_v, empty_vers_v, empty_vers_v, ft.get_sentence_vector("cheval"),
                       ft.get_sentence_vector("huître")])
        v2 = np.stack([empty_vers_v, empty_vers_v, empty_vers_v, ft.get_sentence_vector("cheval"),
                       ft.get_sentence_vector("huître")])
        v3 = np.stack([empty_vers_v, empty_vers_v, ft.get_sentence_vector("cheval"),
                       ft.get_sentence_vector("huître"), ft.get_sentence_vector("Veule sur Saône")])
        v4 = np.stack([empty_vers_v, empty_vers_v, ft.get_sentence_vector("cheval"),
                       ft.get_sentence_vector("huître"), ft.get_sentence_vector("Salamanque")])
        v5 = np.stack([empty_vers_v, ft.get_sentence_vector("cheval"), ft.get_sentence_vector("huître"),
                       ft.get_sentence_vector("Veule sur Saône"), ft.get_sentence_vector("pas tate")])
        v6 = np.stack([empty_vers_v, ft.get_sentence_vector("cheval"), ft.get_sentence_vector("huître"),
                       ft.get_sentence_vector("Veule sur Saône"), ft.get_sentence_vector("J'aime la vie mais aussi P")])
        v7 = np.stack([ft.get_sentence_vector("cheval"), ft.get_sentence_vector("huître"),
                       ft.get_sentence_vector("Veule sur Saône"), ft.get_sentence_vector("pas tate"),
                       ft.get_sentence_vector("on baize")])
        v8 = np.stack([ft.get_sentence_vector("cheval"), ft.get_sentence_vector("huître"),
                       ft.get_sentence_vector("Veule sur Saône"), ft.get_sentence_vector("pas tate"),
                       ft.get_sentence_vector("En sang")])
        v9 = np.stack([ft.get_sentence_vector("huître"), ft.get_sentence_vector("Veule sur Saône"),
                       ft.get_sentence_vector("pas tate"), ft.get_sentence_vector("on baize"),
                       ft.get_sentence_vector("En sang")])
        v10 = np.stack([ft.get_sentence_vector("huître"), ft.get_sentence_vector("Veule sur Saône"),
                        ft.get_sentence_vector("pas tate"), ft.get_sentence_vector("on baize"),
                        ft.get_sentence_vector("on baize")])
        v11 = np.stack([empty_vers_v, empty_vers_v, empty_vers_v, ft.get_sentence_vector("Salamanque"),
                        ft.get_sentence_vector("J'aime la vie mais aussi P")])
        v12 = np.stack([empty_vers_v, empty_vers_v, empty_vers_v, ft.get_sentence_vector("Salamanque"),
                        ft.get_sentence_vector("Veule sur Saône")])
        mat_v_attendue = np.stack([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])
        mat_p, mat_v = self.checheur.liste2matrixes(self.checheur.df2list_idx_train(df_test,
                                                                                    n_0_general=1,
                                                                                    coef_calage=1,
                                                                                    graine=23, shuffle=False),
                                                    df_test, labeliser=False)
        mat_pl, mat_vl, lab = self.checheur.liste2matrixes(self.checheur.df2list_idx_train(df_test,
                                                                                           n_0_general=1,
                                                                                           coef_calage=1,
                                                                                           graine=23, shuffle=False),
                                                           df_test, labeliser=True)
        lab_attendus = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        np.testing.assert_array_almost_equal(mat_v, mat_vl)
        np.testing.assert_array_almost_equal(mat_v, mat_v_attendue)
        np.testing.assert_array_equal(mat_p, mat_pl)
        np.testing.assert_array_equal(mat_p, mat_p_attendue)
        self.assertListEqual(lab, lab_attendus)

    @unittest.skip("334.208s et il marche")
    def test_vers2phon_vect(self):
        ft = fasttext.load_model('cc.fr.300.bin')
        dico_u, dico_m, df_w2p = pd.read_pickle(r"C:\Users\remif\PycharmProjects\PoemesProfonds\data\dicos.pickle")
        ltr2idx, phon2idx, t_x, t_y = pp.chars2idx(df_w2p)
        model_lire = load_model(r"C:\Users\remif\PycharmProjects\PoemesProfonds\models\lecteur\CE1_T12_l10.h5")
        lecteur = lc.Lecteur(t_x, t_y, ltr2idx, phon2idx, dico_u, dico_m,
                             n_brnn1=90, n_h1=80, net=model_lire, blank="_")
        p_vide = ""
        v_vide = ft.get_sentence_vector("/s")
        vers_test = "Nous étions ensemble tout à l'heure à Huruglu"
        p_attendu = "nuzetj§z@s@bl°tutal9RayRygly"
        v_attendu = ft.get_sentence_vector("Nous étions ensemble tout à l'heure à Huruglu")
        v_attendu /= np.linalg.norm(v_attendu)
        p_test, v_test = vers2phon_vect(vers_test, lecteur=lecteur, ft=ft)
        p_empty, v_empty = vers2phon_vect('', lecteur=lecteur, ft=ft)
        self.assertEqual(p_attendu, p_test)
        np.testing.assert_array_almost_equal(v_attendu, v_test)
        self.assertEqual(p_empty, p_vide)
        np.testing.assert_array_equal(v_empty, v_vide)
        self.assertRaises(TypeError, vers2phon_vect, 3, lecteur, ft)

    @unittest.skip("il marche mais il nique la mémoire quand on import trop ft et le modèle")
    def test_vers2matrixes(self):
        ft = fasttext.load_model('cc.fr.300.bin')
        dico_u, dico_m, df_w2p = pd.read_pickle(r"C:\Users\remif\PycharmProjects\PoemesProfonds\data\dicos.pickle")
        ltr2idx, phon2idx, t_x, t_y = pp.chars2idx(df_w2p)
        model_lire = load_model(r"C:\Users\remif\PycharmProjects\PoemesProfonds\models\lecteur\CE1_T12_l10.h5")
        lecteur = lc.Lecteur(t_x, t_y, ltr2idx, phon2idx, dico_u, dico_m,
                             n_brnn1=90, n_h1=80, net=model_lire, blank="_")
        v_vide = ft.get_sentence_vector("/s")
        v1 = "Vive le Théatre !"
        v2 = "J'adore Huruglu, la ville du Soleil"
        v3 = "Depuis la sortie de Yamakasi, je pleure du venin"
        v4 = "Les serpents sont formibles"
        v5 = "Quoi ??? Jardinière !"
        p1, vt1 = vers2phon_vect(v1, lecteur, ft)
        p2, vt2 = vers2phon_vect(v2, lecteur, ft)
        p3, vt3 = vers2phon_vect(v3, lecteur, ft)
        p4, vt4 = vers2phon_vect(v4, lecteur, ft)
        p5, vt5 = vers2phon_vect(v5, lecteur, ft)
        mat_p1_attendue = self.checheur.phonemes2one_hot([p1, p2, p3, p4, p5])
        mat_p_vide_attendue = self.checheur.phonemes2one_hot([''] * 3)
        mat_v1_attendue = np.stack([vt1, vt2, vt3, vt4, vt5])
        mat_v_vide_attendue = np.stack([v_vide] * 3)
        mat_p1, mat_v1 = self.checheur.vers2matrixes([v1, v2, v3, v4, v5], lecteur, ft)
        mat_p_vide, mat_v_vide = self.checheur.vers2matrixes([], lecteur, ft, 3)
        np.testing.assert_array_equal(mat_p1, mat_p1_attendue)
        np.testing.assert_array_almost_equal(mat_v1, mat_v1_attendue)
        np.testing.assert_array_equal(mat_p_vide, mat_p_vide_attendue)
        np.testing.assert_array_almost_equal(mat_v_vide, mat_v_vide_attendue)
        self.assertRaises(ValueError, self.checheur.vers2matrixes, ['', p1, p2, p3, p4, p5], lecteur, ft)


if __name__ == '__main__':
    unittest.main()

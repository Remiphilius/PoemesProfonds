# PoemesProfonds

PoemesProfonds is a project created to write automatically French poetry with neural networks. Two main neural networks are used to do so : a text-to-phoneme converter and a predictor that creates the best sequence of verses from a given single verse or given sequence of verses.

## Table of contents

* [Requirements](#requirements)
* [Text-to-phenomes converter](#text-to-phenomes-converter)
    * [Consonants](#consonants)
    * [Voyels and Semi-voyels](#voyels-and-semi-voyels)
    * [Installation](#installation)
    * [Usage](#usage)
* [Best sequence of verses](#best-sequence-of-verses)
    * [Data](#data)
    * [Neural network](#neural-network)
        * [Inputs](#inputs)
        * [Phonemes](#phonemes)
        * [FastText](#fasttext)
        * [Concatenation](#concatenation)
    * [Writing poetry](#writing-poetry)
    * [Models](#models)
        * [Focused](#focused)
        * [Fast](#fast)
    * [Usage](#usage)
    * [Example](#example)
* [Possible evolutions](#possible-evolutions)
* [References](#referenes)
* [License](#license)

## Requirements

- Python >= 3.5
- Tensorflow 2
- Keras >= 2.3.1
- Java 1.8


## Text-to-phenomes converter

In order to get the phonemes out of the verses from the poetry, a text-to-phonemes converter for French needed to be developped. Some words, especially proper nouns, may appear in poetry but may not in any database. Thus, a model based on neural networks was created so every verse has its phoneme representation.

The data to train the model on was found in [1]. Instead of using the International Phonetic Alphabet (IPA), a substitution alphabet is used as in [1]. It is the one described below:

### Consonants

|IPA symbol            |k|p|l|t|&#641;|f|s|d|&#658;|n|b|v|g|m|z|&#643;|&#626;|&#627;|x|
|----------------------|-|-|-|-|------|-|-|-|------|-|-|-|-|-|-|------|------|------|-|
|Corresponding character|k|p|l|t|R     |f|s|d|Z     |n|b|v|g|m|z|S     |N     |G     |x|

### Voyels and Semi-voyels
|IPA symbol            |a|&#603;|&#596;~|j|o|&#596;|i|&#339;~|&#603;~|e|u|&#593;~|&#601;|&#339;|w|y|&#613;|&oslash;|
|----------------------|-|------|-------|-|-|------|-|-------|-------|-|-|-------|------|------|-|-|------|-|
|Corresponding character|a|E     |�      |j|o|O     |i|1      |5      |e|u|@      |�     |9     |w|y|8     |2|

The model takes as input a word up to 25 lettres long and returns a phonetic transcription up to 21 phonemes long.

The architecture of the model features an attention mechanism [2].

The model has a **99.76% accuracy** on words it was not trained on. It seems to be the best (as of August 2020) in French.

A class **Lecteur** was developped to read the texts. The algorithm uses the phonemes in the data [1] thanks to a mapping in the dictionnary *dico_u*.

However, some words can have several pronounciations (*i.e.* "est" can be read /e/ or /&#603;st/). The algorithm uses a dictionnary mapping the word and its part-of-speech (POS) to the phonemes. This mapping is stored in the dictionnary *dico_m*. The keys are a tuple (word, POS) (*i.e.* {("est", "AUX"): /e/} and {("est", "NOUN"): /&#603;st/}).

Therefore, only words absent of these dictionnaries are read by the model.

### Installation

```python
import preprocessing as pp
from lecture import *
from keras.models import load_model

dico_u, dico_m, df_w2p = pd.read_pickle(os.path.join(".", "data", "dicos.pickle"))
ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
model_lire = load_model(os.path.join(".", "models", "lecteur", "CE1_T12_l10.h5"))
lecteur = Lecteur(Tx, Ty, ltr2idx, phon2idx, dico_u, dico_m, n_brnn1=90, n_h1=80, net=model_lire, blank="_")
```
### Usage

There are three main methods in the class *Lecteur*.

- Method **lire_nn** returns a dictionnary mapping words to phonemes using only the neural network model.

```python
>>> lecteur.lire_nn(["cheval", "abistouque"])
{'cheval': 'S�val', 'abistouque': 'abistuk'}
```
- Method **lire_mots** uses words' POS and the dictionnaries *dico_u* and *dico_m* besides the model to read words. It returns a list containing the phonetic transcriptions of the words.

```python
>>> lecteur.lire_mots(["cheval", "abistouque"])
['S�val', 'abistuk']
```

- Method **lire_vers** features the [French *liaisons*](https://en.wikipedia.org/wiki/Liaison_(French)). The POS is considered while applying the *liaison* or not. For instance, with *les enfants ouvrent* there is no *liaison* between *enfants* (noun) and *ouvrent* (verb). The POS-tagger used is [*StanfordPOSTagger*](https://nlp.stanford.edu/software/tagger.shtml) [6]. Because of it, only a single sentence should be input to the method.

Numbers can also be read thanks to a script broadly inspired by [3].

As this text-to-phonemes converter was developped to read French poetry, the phoneme /&#601;/ is added when a word ends with a consonant sound followed by a mute *e* (except at the end of a verse). This was added thanks to the functions *e_final* and *e_final_tokens* used in the method *lire_vers*. These /&#601;/ are neither present in the dictionnaries *dico_u* and *dico_m* nor in the model.

```python
>>> lecteur.lire_vers("Les trains arrivent en gare de Jarlitude, voies 14 et 97.")
'letR5aRiv�t@gaR�d�ZaRlityd�vwakatORzekatR�v5disEt'
```

## Best sequence of verses

This project was inspired by [4]. The aim of this project is to get a realistic sequence of verses with a neural network. This model scans a sequence of previous verses to get the most likely verse to continue this sequence. Some verses are candidates to be chosen as the best sequel. The model predicts a score for each candidate. Here, it is the probability of the verse to be the sequence's sequel.

Unlike in [4], the neural network reckons a verse as a tuple reprensenting its phonemes (got thanks to the text-to-phonemes converter) and its FastText embedding. FastText [5] is a word embedding representation which can derive a unique vector for a sentence. It also considers the punctuation and it is case-sensitive. This allows the model to make more realistic predictions as it considers these elements.

### Data

As the neural network creates a sequences of verses, these verses need to be stored in a data frame. Each observation is a verse. For the training, the verses should be ordered in the data frame like in the poems. The verse line <img src="https://render.githubusercontent.com/render/math?math=i"> is in its original poem right before the verse line <img src="https://render.githubusercontent.com/render/math?math=i %2B 1"> and right after the one line <img src="https://render.githubusercontent.com/render/math?math=i - 1">.

The data frame needs at least these four variables (their names can be changed while creating an instance of *Chercheur2Vers*):

- vers: raw text of the verse
- phonemes: phonemes of the verse
- id: identifying number of the poem from which the verse is from. (Only needed for training)
- vect: FastText representation of the verse. The models are built to use 300-dimension vectors.

This is an example of the data:

| vers                           | phonemes          | id | vect      |
|--------------------------------|-------------------|----|-----------|
| Que les parfums l�gers de ton air embaum�, | k�lepaRf1leZed�t�nER@bome | 23 | (0.001, ..., 0.03) |
| Que tout ce qu'on entend, l'on voit ou l'on respire, | k�tus2k�n@t@l�vwaul�REspiR | 23 | (0.2, ..., 0.004) |
| Tout dise : Ils ont aim� ! | tudizilz�teme | 23 | (0.052, ..., 0.14) |
| Comme je descendais des Fleuves impassibles, | kOm�Z�des@dEdefl9v�z5pasibl | 55 | (0.092, ..., 0.74) |
| Je ne me sentis plus guid� par les haleurs : | Z�n�m�s@tiplygidepaRleal9R |55 | (0.0001, ..., 0.02) |
| Des Peaux-Rouges criards les avaient pris pour cibles | depoRuZ�kRijaRleavEpRipuRsibl | 55 | (0.096, ..., 0.032) |

In order to have a huge amount of verses, the idea was to get French classical plays. They were got from the website *Th��tre classique* [8] through XML files format.

### Neural network

The aim of the neural network is to compute, for a set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses, the probability that a verse is the real one following this set.

#### Inputs

These verses are modeled by two different matrixes:

- <img src="https://render.githubusercontent.com/render/math?math=P">: one hot encoding representation of the phonemes of the verses. Its shape is <img src="https://render.githubusercontent.com/render/math?math=(s %2B 1, l, n_p %2B 1)">. <img src="https://render.githubusercontent.com/render/math?math=l"> is the size of the longest string of phonemes among the data. <img src="https://render.githubusercontent.com/render/math?math=n_p"> is the number of characters of the phonemes alphabet (38 with the alphabet used here). One is added to <img src="https://render.githubusercontent.com/render/math?math=s"> as the matrix holds the representations of the <img src="https://render.githubusercontent.com/render/math?math=s"> verses of the set and the one of the verse for which we want to compute the probability. One is also added to <img src="https://render.githubusercontent.com/render/math?math=n_p"> as there is always a symbol for a blank.
- <img src="https://render.githubusercontent.com/render/math?math=V">: stack of FastText representations of the verses. Its shape is <img src="https://render.githubusercontent.com/render/math?math=(s %2B 1, d)"> with <img src="https://render.githubusercontent.com/render/math?math=d"> the dimension of the FastText vectors.

#### Phonemes

A gated recurrent unit (GRU) layer is used to compute an embedding for each verse's phonemes. This type of layer was chosen because it considers the order of the phonemes to create the embedding. GRU layers have proven to be efficient while having fewer parameters to train than long short-term memory (LSTM) layers. Its activation is the hyperbolic tangent. The layer turns every of the <img src="https://render.githubusercontent.com/render/math?math=s %2B 1"> rows of the matrix <img src="https://render.githubusercontent.com/render/math?math=P"> from a <img src="https://render.githubusercontent.com/render/math?math=(l, n_p %2B 1)">-shaped matrix to a <img src="https://render.githubusercontent.com/render/math?math=n_e">-dimension vector. This layer creates a <img src="https://render.githubusercontent.com/render/math?math=(s %2B 1, n_e)">-shaped matrix. The last row is extracted from this matrix. So that a new <img src="https://render.githubusercontent.com/render/math?math=(s, n_e)">-shaped matrix <img src="https://render.githubusercontent.com/render/math?math=P_e">, representing the set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses, and a <img src="https://render.githubusercontent.com/render/math?math=n_e">-dimension vector, reprensenting the candidate verse, are created.

A LSTM layer is applied on the matrix <img src="https://render.githubusercontent.com/render/math?math=P_e">. It creates an embedding of the set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses for which the model derives the best sequel. This embedding is a <img src="https://render.githubusercontent.com/render/math?math=n_s">-dimension vector. LSTM was chosen over GRU as it has an additionnal cell state vector which can be useful while creating an embedding considering better enclosing rhymes (*rimes croi�es* and *rimes embrass�es*). Once the set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses is embedded, it has to be concatenated to the embedding of the verse which is a candidate for the sequel. This concatenation returns a <img src="https://render.githubusercontent.com/render/math?math=n_s %2B n_e">-dimension vector which represents the set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses and the candidate verse.

A fully-connected layer with <img src="https://render.githubusercontent.com/render/math?math=n_{d1}"> units is applied to the output of the previous concatenation. Another fully-connected layer with <img src="https://render.githubusercontent.com/render/math?math=n_{d2}"> units derives the ultimate embedding of the phonemes of the <img src="https://render.githubusercontent.com/render/math?math=s"> verses and the candidate verse. A leaky rectified linear unit with &alpha; = 0.2 is used as the activation function for these fully-connected layers. These layers are regularized with a 10% dropout. A batch normalization is also applied on these fully-connected layers to avoid vanishing or exploding gradients.

#### FastText

The only architectures producing realistic results, were the symetric ones. That is to say that for both the phonemes and the FastText representation sides, the output of each layer needs to be of the same size. Thus, each verse's FastText representation needs to be turned into a <img src="https://render.githubusercontent.com/render/math?math=n_e">-dimension vector like the phoneme embedding. Therefore, the matrix <img src="https://render.githubusercontent.com/render/math?math=V"> is turned into a <img src="https://render.githubusercontent.com/render/math?math=(s %2B 1, n_e)"> shaped matrix. A fully-connected layer with <img src="https://render.githubusercontent.com/render/math?math=n_e"> units derives the embedding of the same size as the one of the phonemes. The output of the layer is normalized in order not to have some FastText representations to activate more the next layers's units and thus to be more likely to be picked as a sequel, even though they are not the most realistic ones. Indeed, shorter verses seem to have FastText representations with a norm closer to 1 than the longer verses. Thus, shorter verses were more likely to be picked up by the model. This problem is thus fixed by this procedure. This normalization can be considered as the activation function. A 10% dropout regularization and a batch normalization are applied to the output of this layer.

Similarly to the phonemes' side, the <img src="https://render.githubusercontent.com/render/math?math=(s %2B 1, n_e)"> shaped matrix is split into a <img src="https://render.githubusercontent.com/render/math?math=(s, n_e)"> shaped matrix, called <img src="https://render.githubusercontent.com/render/math?math=V_e">,  and a <img src="https://render.githubusercontent.com/render/math?math=n_e">-dimension vector. <img src="https://render.githubusercontent.com/render/math?math=V_e"> represents the set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses and the stand-alone vector the candidate verse. <img src="https://render.githubusercontent.com/render/math?math=V_e"> is input to a GRU layer. It creates an embedding of <img src="https://render.githubusercontent.com/render/math?math=n_s"> dimensions. The GRU captures a relevant embedding of the theme and the grammatical natures (like gender and number) of the <img src="https://render.githubusercontent.com/render/math?math=s"> verses. The GRU may give more weight to the last verse or understand a pattern between each verse. Unlike the phonemes, side which had a LSTM, the additional cell state was not needed as the the theme and nature should be continuous from a verse to the next one.

Like the phonemes' side, the candidate verse and the <img src="https://render.githubusercontent.com/render/math?math=s"> verses embedding are concatenated into a single <img src="https://render.githubusercontent.com/render/math?math=n_e %2B n_s">-dimension vector. It goes through two fully-connected layers with respectively <img src="https://render.githubusercontent.com/render/math?math=n_{d1}"> and <img src="https://render.githubusercontent.com/render/math?math=n_{d2}"> units. Each layer has a leaky rectifer linear unit with &alpha; = 0.2. They are regularized with a 10% dropout. A batch normalization is also applied to booth outputs.

#### Concatenation

The embeddings from both the phonemes and the FastText sides are concatenated. This derives a <img src="https://render.githubusercontent.com/render/math?math=2 n_{d2}">-dimension vector. This vector is input to a fully connected layer with <img src="https://render.githubusercontent.com/render/math?math=n_f"> units. Its activation function is again a leaky rectifier linear unit with &alpha; = 0.2. Again, it is regularized with a 10% dropout. A batch normalization is applied to the output of this layer. Adding more layers here was tried, but it was not as efficient as adding a layer to the phoneme and FastText sides.

This <img src="https://render.githubusercontent.com/render/math?math=n_f">-dimension vector feeds the last layer which has a single unit and is fully-connected. Its activation function is the sigmoid. Thus the number is the probability of the candidate verse to be the sequel of the <img src="https://render.githubusercontent.com/render/math?math=s"> verses.

Bellow, the diagram of the architecture generated with [netron](https://github.com/lutzroeder/Netron):

![](archi.png)


### Writing poetry

In order to generate enclosing rhymes, the poem written will not keep only the most likely verse after having applied the neural network model (greedy search). Similary to machine translation, a **beam search** algorithm is used here to keep the <img src="https://render.githubusercontent.com/render/math?math=k"> best sequences of verses which were derived by the model. Then, the model uses these <img src="https://render.githubusercontent.com/render/math?math=k"> sequences as the set of <img src="https://render.githubusercontent.com/render/math?math=s"> verses and compute the best sequel for each of them. The code is inspired from [7].

### Models

There are two models available to create the sequences. The first one is the most precise but it about twelve times longer to compute than the second one which produced slighly less good results. For both of them: <img src="https://render.githubusercontent.com/render/math?math=s"> = 8, <img src="https://render.githubusercontent.com/render/math?math=n_p"> = 38, <img src="https://render.githubusercontent.com/render/math?math=l"> = 51.

#### Focused

This model has 2,982,751 parameters.

- <img src="https://render.githubusercontent.com/render/math?math=n_e"> = 300
- <img src="https://render.githubusercontent.com/render/math?math=n_s"> = 425
- <img src="https://render.githubusercontent.com/render/math?math=n_{d1}"> = 250
- <img src="https://render.githubusercontent.com/render/math?math=n_{d2}"> = 100
- <img src="https://render.githubusercontent.com/render/math?math=n_f"> = 50

#### Fast

This model has 240,231 parameters.

- <img src="https://render.githubusercontent.com/render/math?math=n_e"> = 50
- <img src="https://render.githubusercontent.com/render/math?math=n_s"> = 140
- <img src="https://render.githubusercontent.com/render/math?math=n_{d1}"> = 50
- <img src="https://render.githubusercontent.com/render/math?math=n_{d2}"> = 30
- <img src="https://render.githubusercontent.com/render/math?math=n_f"> = 25

### Usage

The user have to write at least one verse by one self. The model writes a poem from it.

```python
import preprocessing as pp
from lecture import *
from chercheur2vers import *
from keras.models import load_model
import fasttext.util

dico_u, dico_m, df_w2p = pd.read_pickle(os.path.join(".", "data", "dicos.pickle"))
ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
model_lire = load_model(os.path.join(".", "models", "lecteur", "lecteur_mdl.h5"))
lecteur = Lecteur(Tx, Ty, ltr2idx, phon2idx, dico_u, dico_m, n_brnn1=90, n_h1=80, net=model_lire, blank="_")

ft = fasttext.load_model(os.path.join("models", "FastText", 'cc.fr.300.bin'))

vers = pd.read_pickle(os.path.join("data", "vers.pkl"))
_, vers_test = split_train_dev(vers, test_size=0.03)

model_chercher = load_model(os.path.join(".", "models", "chercher_vers", "fast.h5"))
checheur = Chercheur2Vers(t_p=50, p2idx=phon2idx, net=model_chercher, n_antecedant_vers=8)

poem = checheur.beam_search_write(["Apr�s cette peine, emprisonn� par l'ennui,",
                                   "Je suis mon cher ami tr�s heureux de te voir."],
                                  df=vers_test, vers_suivants=5, k=5, batch_size=512, split=100
                                  lecteur=lecteur, ft=ft)
```
Two parameters are important for the quality of the poem generated and the speed of execution:

- **test_size**: amount of the poems in the data to pick the verses from to write the sequel
- **k**: beam width

If there is any memory error, it is possible to divide the size of the input matrixes thanks to the parameter **split**. The higher it is, the less memory is required.

It is also possible to export formated inputs from verses to run the neural networks on a GPU or TPU (like colab) elsewhere:

1. Exporting the inputs

```python
phoneme_input, fasttext_input = checheur.vers2matrixes(["Apr�s cette peine emprisonn� par l'ennui,",
                                                        "Je suis mon cher ami tr�s heureux de te voir."],
                                                        lecteur, ft, len_output=8)
```

2. Using them

```python
poem = checheur.beam_search_write(liste_vers=["Apr�s cette peine emprisonn� par l'ennui,",
                                              "Je suis mon cher ami tr�s heureux de te voir."],
                                              df=vers_test, vers_suivants=7, k=3, split=100, batch_size=512,
                                              mphon_prec=phoneme_input, mvect_prec=fasttext_input)
```

### Example

*Harry* :
Apr�s cette peine, emprisonn� par l'ennui,
Je suis mon cher ami tr�s heureux de te voir.
J'en attends peu de chose, et br�le de le voir.
Pass�s aupr�s de toi, blonde soeur d'Oph�lie,
Que se borne en effet le roman de ma vie.
Vous aviez d�sol� les pays d'alentour,
�gare votre coeur, vous s�duit en ce jour.

*Dauphin* :
Apr�s cette peine, emprisonn� par l'ennui,
Je suis mon cher ami tr�s heureux de te voir.
J'en attends peu de chose, et br�le de le voir.
Pass�s aupr�s de toi, blonde soeur d'Oph�lie,
Que se borne en effet le roman de ma vie.
Vous aviez d�sol� les pays d'alentour,
Que tous les Citoyens vous rendent tour � tour.

*�lu* :
Apr�s cette peine, emprisonn� par l'ennui,
Je suis mon cher ami tr�s heureux de te voir.
J'en attends peu de chose, et br�le de le voir.
Pass�s aupr�s de toi, blonde soeur d'Oph�lie,
Que se borne en effet le roman de ma vie.
Vous aviez d�sol� les pays d'alentour,
Qu'un peu de vanit� se sent dans vos atours.

*Harry* is the third most likely, *Dauphin* the second and *�lu* is the most likely. There are three results displayed thanks to the beam seach (when k = 2, *Harry* is not displayed as it is not computed). These poems was got with **k** = 5, **vers_suivants** = 5 and the model **fast**.

## Possible evolutions

A third neural network which can generate its own verses can be developped so this algorithm could write poetry all by itself. It can be done with a generative adversarial network or with reinforcement learning.

The text-to-phonemes converter can be used in various applications like text-to-speech applications.

The whole algorithm can also be used to create songs lyrics.

## References
[1] [New, Boris, Christophe Pallier, Ludovic Ferrand, and Rafael Matos. 2001. "Une Base de Donn�es Lexicales Du Fran�ais Contemporain Sur Internet: LEXIQUE" L'Ann�e Psychologique 101 (3): 447-462](https://chrplr.github.io/openlexicon/datasets-info/Lexique382/New%20et%20al.%20-%202001%20-%20Une%20base%20de%20donn%C3%A9es%20lexicales%20du%20fran%C3%A7ais%20contempo.pdf)

[2] [Vaswani, A., et al.: Attention is all you need. arXiv (2017). arXiv:1706.03762](https://arxiv.org/pdf/1706.03762.pdf)

[3] [Michel Claveau. 2004. "Traduction nombre => texte" from mclaveau.com](http://mclaveau.com/ress/python/trad.py)

[4] [E Malmi, P Takala, H Toivonen, T Raiko, A. Gionis. 2016. DopeLearning: A Computational Approach to Rap Lyrics Generation. arXiv preprint arXiv:1505.04771](https://arxiv.org/pdf/1505.04771.pdf)

[5] [Tomas Mikolov, Edouard Grave, Piotr Bojanowski, Christian Puhrsch, and Armand Joulin.   Advances in pre-training distributed word representations.  InProceedings of the Eleventh International Conference on LanguageResources and Evaluation (LREC-2018), 2018.](https://arxiv.org/pdf/1712.09405.pdf)

[6] [Kristina Toutanova, Dan Klein, Christopher Manning, and Yoram Singer. 2003. Feature-Rich Part-of-Speech Tagging with a Cyclic Dependency Network. In Proceedings of HLT-NAACL 2003, pp. 252-259.](https://nlp.stanford.edu/~manning/papers/tagging.pdf)

[7] [Jason Brownlee. January 5 2018. "How to Implement a Beam Search Decoder for Natural Language Processing" from machinelearningmastery.com](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/)

[8] [Th��tre classique](http://theatre-classique.fr/pages/programmes/PageEdition.php)

## License

� R�mi Desmarescaux, Ryan Ugolini

Licensed under the [MIT License](LICENSE).

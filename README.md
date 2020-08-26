---
output:
  html_document: default
  pdf_document: default
---
# PoemesProfonds

## Converter text to phenomes

In order to get the phonemes of the verses from the poetry, a model converting word to phonemes in French was developped, especially for proper nouns. The data to train the model on was found in [1]. Instead of using the International Phonetic Alphabet (IPA), the same substitution alphabet is used as in [1] as described below. 

### Consonants

|IPA symbol            |k|p|l|t|&#641;|f|s|d|&#658;|n|b|v|g|m|z|&#643;|&#626;|&#627;|x|
|----------------------|-|-|-|-|------|-|-|-|------|-|-|-|-|-|-|------|------|------|-|
|Corresponding character|k|p|l|t|R     |f|s|d|Z     |n|b|v|g|m|z|S     |N     |G     |x|

### Voyels and Semi-voyels
|IPA symbol            |a|&#603;|&#596;~|j|o|&#596;|i|&#339;~|&#603;~|e|u|&#593;~|&#601;|&#339;|w|y|&#613;|&oslash;|
|----------------------|-|------|-------|-|-|------|-|-------|-------|-|-|-------|------|------|-|-|------|-|
|Corresponding character|a|E     |§      |j|o|O     |i|1      |5      |e|u|@      |°     |9     |w|y|8     |2|

The model can take as input a word up to 25 lettres long and return a phonetic transcription up to 21 phonemes long.

The architecture of the model features an attention mechanism as seen in [2].

The model has a **99.76% accuracy** on words it was not trained on which seem to be the best (as of August 2020) in French.

A class **Lecteur** was developped to read the texts. It uses dictionnaries mapping words to phonemes. The words already present in the data used to train the model are not read by the model. The algorithm uses the phonemes in the data. This mapping in the dictionnary *dico_u*.

However, some words can have several pronounciations (*i.e.* "est" can be read /e/ or /&#603;st/). The algorithm uses a dictionnary mapping the word and its part-of-speech (POS) to the phonemes. This mapping is stored in the dictionnary *dico_m*. The keys are a tuple (word, POS).

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

Method **lire_nn** returns a dictionnary mapping words to phonemes using only the neural network model.

```python
>>> lecteur.lire_nn(["cheval", "abistouque"])
{'cheval': 'S°val', 'abistouque': 'abistuk'}
```
Method **lire_mots** uses words POS and the dictionnaries *dico_u* and *dico_m* besides the model to read words. It returns a list containing the phonetic transcriptions of the words.

```python
>>> lecteur.lire_mots(["cheval", "abistouque"])
['S°val', 'abistuk']
```

As this text-to-phonemes converter was developped to read French poetry, the phoneme /&#601;/ is added when a word ends with a consonant sound followed by a mute "e" (except at the end of a verse). This was added thanks to the functions *e_final* and *e_final_tokens* used in the method *lire_vers*. Theses /&#601;/ are nor present in the dictionnaries *dico_u* and *dico_m* neither in the model.

Method **lire_vers** also features the [French *liaisons*](https://en.wikipedia.org/wiki/Liaison_(French)). The POS is considered while applying the *liaison* or not. Numbers can also be read.

```python
>>> lecteur.lire_vers("Les trains arrivent en gare de Jarlitude sur les voies 14 et 97.")
'letR5aRiv°t@gaR°d°ZaRlityd°syRlevwakatORzekatR°v5disEt'
```

## References
[1] [New, Boris, Christophe Pallier, Ludovic Ferrand, and Rafael Matos. 2001. "Une Base de Données Lexicales Du Français Contemporain Sur Internet: LEXIQUE" L'Année Psychologique 101 (3): 447-462](https://chrplr.github.io/openlexicon/datasets-info/Lexique382/New%20et%20al.%20-%202001%20-%20Une%20base%20de%20donn%C3%A9es%20lexicales%20du%20fran%C3%A7ais%20contempo.pdf)

[2] [Vaswani, A., et al.: Attention is all you need. arXiv (2017). arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## License

# PoemesProfonds

## Converter text to phenomes

In order to get the phonemes of the verses from the poetry, a model converting word to phonemes was developped, especially for proper nouns. The data to train the model on was found in [1]. Instead of using the International Phonetic Alphabet (IPA), the same substitution alphabet is used as in [1] as described below. 

### Consonants

|IPA symbol            |k|p|l|t|&#641;|f|s|d|&#658;|n|b|v|g|m|z|&#643;|&#626;|&#627;|x|
|----------------------|-|-|-|-|------|-|-|-|------|-|-|-|-|-|-|------|------|------|-|
|Corresponding character|k|p|l|t|R     |f|s|d|Z     |n|b|v|g|m|z|S     |N     |G     |x|

### Voyels and Semi-voyels
|IPA symbol            |a|&#603;|&#596;~|j|o|&#596;|i|&#339;~|&#603;~|e|u|&#593;~|&#601;|&#339;|w|y|&#613;|&oslash;|
|----------------------|-|------|-------|-|-|------|-|-------|-------|-|-|-------|------|------|-|-|------|-|
|Corresponding character|a|E     |§      |j|o|O     |i|1      |5      |e|u|@      |°     |9     |w|y|8     |2|

The model can take as input a word up to 25 lettres long and return a phonetic transcription up to 21 phonemes long.

The model has a **99.76% accuracy** on words it was not trained on.

The aim of this model is to read words part of poetry. Thus, it was built to add the phoneme /&#601;/ when a word ends with a consonant sound followed by a mute "e" (except at the end of a verse).

```python
import preprocessing as pp
from lecture import *
from keras.models import load_model

dico_u, dico_m, df_w2p = pd.read_pickle(os.path.join(".", "data", "dicos.pickle"))
ltr2idx, phon2idx, Tx, Ty = pp.chars2idx(df_w2p)
model_lire = load_model(os.path.join(".", "models", "lecteur", "CE1_T12_l10.h5"))
lecteur = Lecteur(Tx, Ty, ltr2idx, phon2idx, dico_u, dico_m, n_brnn1=90, n_h1=80, net=model_lire, blank="_")
```

```python
>>> lecteur.lire_nn(["cheval", "abistouque"])
{'cheval': 'S°val', 'abistouque': 'abistuk'}
```

```python
>>> lecteur.lire_mots(["cheval", "abistouque"])
['S°val', 'abistuk']
```

```python
>>> lecteur.lire_vers("Ils sont arrivés en Allemagne.")
'ils§taRivez@nal°maN'
```

## References
[1] [New, Boris, Christophe Pallier, Ludovic Ferrand, and Rafael Matos. 2001. "Une Base de Données Lexicales Du Français Contemporain Sur Internet: LEXIQUE" L'Année Psychologique 101 (3): 447-462][https://chrplr.github.io/openlexicon/datasets-info/Lexique382/New%20et%20al.%20-%202001%20-%20Une%20base%20de%20donn%C3%A9es%20lexicales%20du%20fran%C3%A7ais%20contempo.pdf]

[2] [Vaswani, A., et al.: Attention is all you need. arXiv (2017). arXiv:1706.03762][https://arxiv.org/abs/1706.03762]

## License

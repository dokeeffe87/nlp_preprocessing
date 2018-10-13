from __future__ import division

import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import pyLDAvis
import pyLDAvis.gensim

from nltk.corpus import stopwords

from langdetect import detect

from tqdm import tqdm


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en', disable=['parser', 'ner'])


def remove_non_english(df, text_col):
    """
    Remove non english language
    :param df: DataFrame with text
    :param text_col: Column name of text in df
    :return: DataFrame with only english text
    """
    def find_lang(x):
        try:
            return detect(x)
        except:
            return 'none'

    tqdm.pandas()
    df['language'] = df[text_col].progress_apply(find_lang)

    return df.loc[df.language == 'en', :]


def remove_non_custom_language(df, text_col, lang_list):
    """
    Removes all text except for desired languages
    :param df: DataFrame with text
    :param text_col: Column name of text in df
    :param lang_list: List of languages to keep.  Supported languages are list here: https://pypi.org/project/langdetect/
    :return: DataFrame with only selected language text
    """
    def find_lang(x):
        try:
            return detect(x)
        except:
            return 'none'

    tqdm.pandas()
    df['language'] = df[text_col].progress_apply(find_lang)

    return df.loc[df.language.isin(lang_list)]


def pre_process_regex(df, text_col):
    """
    Removes newline characters and single quotes
    :param df: DataFrame with text
    :param text_col: Column name of text in df
    :return: List with text with newline characters and single quotes removed
    """
    # Convert to list
    data_re = df[text_col].values.tolist()

    # Remove newline characters
    print('Removing newline characters.')
    data_re = [re.sub('\s+', ' ', sent) for sent in tqdm(data_re)]

    # Remove single quotes
    print('Removing single quotes')
    data_re = [re.sub("\'", "", sent) for sent in tqdm(data_re)]

    return data_re


def sent_to_words(data_list):
    """
    Tokenize words and remove punctuation.
    :param data_list: List containing sentences to process
    :return: List of tokenized sentences
    """
    # Tokenize words and remove punctuation
    for sentence in tqdm(data_list):
        yield (gensim.utils.simple_preprocessing(str(sentence), deacc=True))


def remove_stopwords(texts):
    """
    Removes stopwords from input text
    :param texts: Text to process
    :return: List of Lists.  Each document becomes a list of processed words without stopwords
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in tqdm(texts)]


def make_bigrams_trigrams(texts, min_count=5, threshold=100):
    """
    Makes bigrams and trigrams out of input texts
    :param texts: Input texts
    :param min_count: min_count for bigram model.  The higher this is, the harder it is to make bigrams
    :param threshold: threshold for bigram model.  The higher this is, the harder it is to make bigrams
    :return: List of bigrams and list of trigrams
    """
    bigram = gensim.models.Phrases(texts, min_count=min_count, threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return [bigram_mod[doc] for doc in tqdm(texts)], [trigram_mod[bigram_mod[doc]] for doc in tqdm(texts)]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ABV']):
    """
    Lemmatizes input text
    :param texts: Text to be lemmatized
    :param allowed_postags: Allowed postags to be used in spacey to lemmatize
    :return: Lemmatized text
    """
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out


def create_dictionary(data_lemmatized):
    """
    Create the dictionary to use for topic modeling
    :param data_lemmatized: Lemmatized texts
    :return: Dictionary
    """
    return corpora.Dictionary(data_lemmatized)


def create_corpus(id2word, data_lemmatized):
    """
    Create the corpus for topic modeling
    :param id2word: The mapping of ids to words in the dictionary
    :param data_lemmatized: Lemmatized texts
    :return: A corpus
    """
    return [id2word.doc2bow(text) for text in tqdm(data_lemmatized)]


def lda_preprocess(df: object, text_col: object, lang_list: list, min_count=5, threshold=100, trigrams=False) -> object:
    """
    Applies the functions above to preprocess text data for
    :param df: A DataFrame which contains the text you want to process
    :param text_col: The column name in df which contains the text
    :param lang_list: List of languages you want to consider. See https://pypi.org/project/langdetect/ for supported languages
    :param min_count: min_count for bigram model.  The higher this is, the harder it is to make bigrams
    :param threshold: threshold for bigram model.  The higher this is, the harder it is to make bigrams
    :param trigrams: True if you want to lemmatize on trigrams, default is False
    :return: dictionary, corpus, and lemmatized text data
    """
    print("Begin LDA preprocessing")

    print("Detecting languages")
    df_lang = remove_non_custom_language(df, text_col, lang_list)
    print('Language detection complete. Detected and retained languages: {0}'.format(df_lang['language'].unique().tolist()))

    print("Begin regex preprocessing")
    data = pre_process_regex(df_lang, text_col)
    print('End regex preprocessing')

    print('Begin sentence preprocessing')
    data_words = list(sent_to_words(data))
    print('End sentence preprocessing')

    print('Begin stopword removal')
    data_words_nostops = remove_stopwords(data_words)
    print('end remove stopwords')

    # make bigrams and trigrams.
    print('Building bigrams and trigrams')
    data_words_bigrams, data_words_trigrams = make_bigrams_trigrams(data_words_nostops, min_count=min_count, threshold=threshold)

    # Lemmatize:
    print("Begin lemmatization")
    data_lemmatized = 

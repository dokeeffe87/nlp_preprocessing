import os
import gzip
import re
import pandas as pd
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import urllib
import zipfile
import collections
import tensorflow as tf
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import pyLDAvis
import pyLDAvis.gensim

import nltk

from langdetect import detect, DetectorFactory

from tqdm import tqdm

from pprint import pprint

import warnings

import sklearn.preprocessing

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Dictionary of supported languages (for stopwords), from their ISO codes to english names.
supported_languages = {'sv': 'swedish',
                       'da': 'danish',
                       'id': 'indonesian',
                       'az': 'azerbaijani',
                       'fi': 'finnish',
                       'no': 'norwegian',
                       'de': 'german',
                       'kk': 'kazakh',
                       'nl': 'dutch',
                       'fr': 'french',
                       'ru': 'russian',
                       'el': 'greek',
                       'es': 'spanish',
                       'ro': 'romanian',
                       'en': 'english',
                       'pt': 'portuguese',
                       'tr': 'turkish',
                       'ar': 'arabic',
                       'ne': 'nepali',
                       'hu': 'hungarian',
                       'it': 'italian'}

# The language detection algorithm is non-deterministic on short text. We can enforce reproducibility by setting the seed
DetectorFactory.seed = 0


def remove_non_english(df, text_col):
    """
    Remove non english language
    :param df: DataFrame with text
    :param text_col: Column name of text in df
    :return: DataFrame with only english text
    """
    # TODO: Test this with changes.
    # def find_lang(x):
    #     try:
    #         return detect(x)
    #     except:
    #         return 'none'

    tqdm.pandas()
    df['language'] = df[text_col].progress_apply(find_lang)

    return df.loc[df.language == 'en', :]


def find_lang(x):
    """
    Detect language of input string.
    :param x: String on which to detect language.  It can be any size, but this will generally work much better with longer sentences
    :return: The detected language, or the string none if language can't be identified
    """
    try:
        return detect(x)
    except:
        return 'none'


def remove_non_custom_language(df, text_col, lang_list):
    """
    Removes all text except for desired languages
    :param df: DataFrame with text
    :param text_col: Column name of text in df
    :param lang_list: List of languages to keep.  Supported languages are list here: https://pypi.org/project/langdetect/
    :return: DataFrame with only selected language text
    """

    tqdm.pandas()
    df['language'] = df[text_col].progress_apply(find_lang)

    print('Languages detected: {0}'.format(df['language'].unique()))

    if lang_list:
        return df.loc[df.language.isin(lang_list)]
    else:
        return df


def download_file(filename, url):
    """
    Download a file if it exists at a given url
    :param filename: The name of the file you are trying to download
    :param url: The url where you want to download the file from
    :return: The name of the file which should be in your current working directory
    """
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if not statinfo:
        print('File not found')
    return filename


def read_data_from_zip(filename):
    """
    Get text data out of a supplied zip file
    :param filename: Name of the zip file to use
    :return: The text data as a list of a list unicode strings. The list of list structure is to make the returned object immediately compatible with dictionary creation.
    """
    # TODO: test if this is still necessary in python3.
    with zipfile.ZipFile(filename) as f:
        # Tensorflow compat adds compatibility between python 2 and 3 functions
        # as_str returns a unicode string from the argument
        # namelist returns a list of archiv memebers by name from the zip file
        data = [tf.compat.as_str(f.read(f.namelist()[0])).split()]
    return data


def trim_vocabulary(vocab, vocab_size):
    """
    A function to trim down a large vocabulary to the vocab_size most common words
    :param vocab: The vocabulary. Assumes a list of lists, each list is a document and the sublist is the vocabulary for that document
    :param vocab_size: The number of most common words to retain
    :return: A list of lists. Each list contains another list with the vocabulary for each document trimmed down to the vocab_size most common words
    """
    trimmed_vocabs = []
    for word_list in vocab:
        counter = collections.Counter(word_list)
        most_common = sorted(counter, key=counter.get, reverse=True)[:vocab_size]
        trimmed_vocabs.append(most_common)
    return trimmed_vocabs


def get_stopwords(lang_list):
    """
    Gets the list of stopwords for the specified languages.  Currently only supports the languages listed in the supported_languages
    dictionary.  If you pass it a language not currently supported, it will be skipped.
    :param lang_list: List of ISO codes for required languages
    :return: List of stopwords
    """
    words_to_return = []
    for lang in lang_list:
        try:
            lang_name = supported_languages[lang]
            words_to_return += nltk.corpus.stopwords.words(lang_name)
        except KeyError:
            print('Language {0} not currently supported for stopword removal. Skipping it'.format(lang))

    return words_to_return


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


def sent_to_words(data_list, use_gensim=True):
    """
    Tokenize words and remove punctuation.
    :param data_list: List containing sentences to process
    :param use_gensim: Boolean. Whether or not to use gensim for tokenization. Otheriwse will use simple nltk word_tokenize
    :return: List of tokenized sentences
    """
    # TODO: Make punctuation removal an option.
    # Tokenize words and remove punctuation
    for sentence in tqdm(data_list):
        if use_gensim:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))
        else:
            yield (nltk.tokenize.word_tokenize(str(sentence)))


def remove_stopwords(texts, stop_words):
    """
    Removes stopwords from input text
    :param texts: Text to process
    :param stop_words: List of stop words to filter.
    :return: List of Lists.  Each document becomes a list of processed words without stopwords
    """
    return [[word for word in doc if word not in stop_words] for doc in tqdm(texts)]


def remove_punctuation_string(text_str, punc_):
    """
    Removes characters from punctuation list. Returns a string
    :param text_str: String with text to clean
    :param punc_: List of characters to remove
    :return: Text string with elements of punc_ removed
    """
    punc_ = "".join(punc_)
    unicode_line = text_str.translate({ord(c): None for c in punc_})

    return unicode_line


def remove_stopwords_string(text_str, stop_words):
    """
    Removed words from input text. Returns a string
    :param text_str: String with text to clean
    :param stop_words: List of words to remove
    :return: String with elements of stop_words removed
    """
    text_str = text_str.lower()
    text_list = text_str.split(' ')
    text_list = [x for x in text_list if x not in stop_words]
    text_str_return = " ".join(text_list)

    return text_str_return


def make_bigrams_trigrams(texts, min_count=5, threshold=100):
    """
    Makes bigrams and trigrams out of input texts
    :param texts: Input texts
    :param min_count: min_count for bigram model.  The higher this is, the harder it is to make bigrams
    :param threshold: threshold for bigram model.  The higher this is, the harder it is to make bigrams
    :return: List of bigrams and list of trigrams
    """
    # TODO: Getting a lot of user warnings here.  Figure out why.
    bigram = gensim.models.Phrases(texts, min_count=min_count, threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return [bigram_mod[doc] for doc in tqdm(texts)], [trigram_mod[bigram_mod[doc]] for doc in tqdm(texts)]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ABV']):
    """
    Lemmatizes input text with allowed part of speech tags
    :param texts: Text to be lemmatized
    :param allowed_postags: Allowed postags to be used in spacey to lemmatize
    :return: Lemmatized text
    """
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out


def create_dictionary(data_lemmatized, args={}):
    """
    Create the dictionary to use for topic modeling
    :param data_lemmatized: Lemmatized texts
    :param args: Dictionary of parameters to send to gensim Dictionary
    :return: Dictionary. Note this object has  built in method to get the reversed dictionary
    """
    return corpora.Dictionary(data_lemmatized, **args)


def create_reversed_dictionary(dictionary_):
    """
    Create the reversed dictionary to look up ids by word
    :param dictionary_: The dictionary created from the vocabulary
    "return: a dictionary object containing the reverse value-key pairs of dictionary_
    """
    if type(dictionary_).__name__ == 'Dictionary':
        # If the dictionary is a gensim dictionary object, use the built in methond.
        print('Gensim Dictionary type detected')
        return dictionary_.token2id
    elif type(dictionary_).__name__ == 'dict':
        print('Input dictionary of type python dict')
        # If the dictionary is a regular python dictionary, return the reversed dictionary
        return dict(zip(dictionary_.values(), dictionary_.keys()))
    else:
        print('Unknown dictionary type')
        return None


def build_data_for_keras_skipgram(vocabulary, vocabulary_size):
    """
    TODO: Generalize to work with multiple documents.
    To use the built in Keras skipgram method, we need a dictionary where the index of the word represents it's rank.  It won't work otherwise. So we can't directly use the Gensim dictionary object. Note that skipgram assumes that the 0th index word represents non-word and is skipped
    :param vocabulary: Vocabulary. A list of tokenized words.
    :param vocabulary_size: The integer size of most frequent words to retain
    :return: The dictionary (index -> word) and reversed dictionary (word -> index)
    """
    count = [["UNK", -1]]
    count.extend(collections.Counter(vocabulary).most_common(vocabulary_size - 1))
    reversed_dictionary = {}
    for word, _ in count:
        reversed_dictionary[word] = len(reversed_dictionary)
    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))
    return dictionary, reversed_dictionary


def make_sequence_of_indices_from_vocabulary(vocabulary, reversed_dictionary):
    """
    A function to create a sequence of indices from a given vocabulary.  This is to preserve the sequence of words in the documents for context modeling.
    :param vocabulary: The vocabulary of words used per document.  Should be a list of lists, one list per document and the sublists should be tokenized words in the document with preserved order
    :param reversed_dictionary: The reversed_dictionary (word -> index) created from the either the original vocabulary or a trimmed vocabulary.
    """
    sequence_of_indices = []
    # Keep track of the number of words we have skipped over by trimming the vocabulary
    unk_count = []
    for doc_vocab in vocabulary:
        doc_sequence_of_indices = []
        doc_unk_count = 0
        for word in doc_vocab:
            if word in reversed_dictionary:
                index = reversed_dictionary[word]
            else:
                index = reversed_dictionary['UNK']
                doc_unk_count += 1
            doc_sequence_of_indices.append(index)
        sequence_of_indices.append(doc_sequence_of_indices)
        unk_count.append(doc_unk_count)
    return sequence_of_indices, unk_count


def create_corpus(id2word, data_lemmatized):
    """
    Create the corpus for topic modeling
    :param id2word: The mapping of ids to words in the dictionary
    :param data_lemmatized: Lemmatized texts
    :return: A corpus
    """
    return [id2word.doc2bow(text) for text in tqdm(data_lemmatized)]


def lda_preprocess(df, text_col, lang_list=None, min_count=5, threshold=100, trigrams=False, custom_remove=None, override_langdetect=False, override_language=None, use_gensim_tokenize=True, remove_single_characters=False, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ABV']):
    """
    Applies the functions above to preprocess text data for
    :param df: A DataFrame which contains the text you want to process
    :param text_col: The column name in df which contains the text
    :param lang_list: List of languages you want to consider. See https://pypi.org/project/langdetect/ for supported languages
    :param min_count: min_count for bigram model.  The higher this is, the harder it is to make bigrams
    :param threshold: threshold for bigram model.  The higher this is, the harder it is to make bigrams
    :param trigrams: True if you want to lemmatize on trigrams, default is False
    :param custom_remove: List of additional words to remove (treat as stopwords)
    :param override_langdetect: Boolean. Set True if you want to skip language detection. You might want to do this if you already know all of your text is in one language (which you can provide to
                                the override_language argument. Supported languages here: https://pypi.org/project/langdetect/), or if your documents are small, which can lead to language
                                misclassification
    :param override_language: The language you want to use as the default language. See list of supported language (https://pypi.org/project/langdetect/)
    :param use_gensim_tokenize: Boolean. Whether or not to use gensim simple_preprocessing for tokenization. This usually removes non-letter characters. If you want to keep some numbers, set this
                                argument to False and tokenization will be done with simple nltk word_tokenize
    :param remove_single_characters: Boolean. Set true if you want to ensure that there are no single characters left over in your processed text data. This is only really an issue if you set
                                     use_gensim_tokenize to False
    :param allowed_postags: List of allowed part of speech tags to allow when lemmatizing. This will keep only words with the input tags.  Supported tags can be found here (look for POS column in the
                            first table): https://spacy.io/usage/linguistic-features
    :return: dictionary, corpus, and lemmatized text data
    """
    # TODO: Add support for stemming.
    # TODO: Add support for automatic number to word conversion.
    print("Begin LDA preprocessing")

    if not override_langdetect:
        print("Detecting languages")
        df_lang = remove_non_custom_language(df, text_col, lang_list)
        if lang_list is not None:
            print('Language detection complete. Detected and retained languages: {0}'.format(df_lang['language'].unique().tolist()))
        else:
            print('Language detection complete. All languages retained.')
    elif override_langdetect and override_language is not None:
        print('Language detection skipped. Attempting to use provided language: {0}'.format(override_language))
        df['language'] = override_language
        df_lang = df.copy()
    else:
        print('Language detection skipped. Nothing provided to override_language argument, attempting to assume English text')
        df['language'] = 'en'
        df_lang = df.copy()

    print("Begin regex preprocessing")
    data = pre_process_regex(df_lang, text_col)
    print('End regex preprocessing')

    print('Begin sentence preprocessing')
    if use_gensim_tokenize:
        data_words = list(sent_to_words(data))
    else:
        data_words = sent_to_words(data, use_gensim=False)
    print('End sentence preprocessing')

    print('Begin stopword removal')
    if lang_list is not None:
        print('Generating stopword list for specified languages: {0}'.format(df_lang['language'].unique().tolist()))
        stop_words = get_stopwords(lang_list)
    else:
        print('Generating stopword list for specified languages: {0}'.format(df_lang['language'].unique().tolist()))
        stop_words = get_stopwords(df_lang['language'].unique().tolist())
    if custom_remove is not None:
        print("Adding custom words for removal")
        stop_words += custom_remove
        data_words_nostops = remove_stopwords(data_words, stop_words=stop_words)
    else:
        data_words_nostops = remove_stopwords(data_words, stop_words=stop_words)
    print('end remove stopwords')
    # make bigrams and trigrams.
    print('Building bigrams and trigrams')
    data_words_bigrams, data_words_trigrams = make_bigrams_trigrams(data_words_nostops, min_count=min_count, threshold=threshold)

    # Lemmatize:
    print("Begin lemmatization")
    if trigrams:
        data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=allowed_postags)
    else:
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=allowed_postags)
    print('End lemmatization')

    if remove_single_characters:
        data_lemmatized = [[w for w in sentence if len(w) > 1] for sentence in data_lemmatized]

    print('Begin creating dictionary')
    id2word = create_dictionary(data_lemmatized)
    print('End creating dictionary')

    print('Being creating corpus')
    corpus = create_corpus(id2word, data_lemmatized)
    print('End creating corpus')

    print('End LDA preprocessing')

    return id2word, corpus, data_lemmatized


def compute_coherence_values(dictionary, corpus, texts, limit, window_size, coherence='c_v', start=2, step=3, mallet_path=None, args={}):
    """
    Compute the Cv coherence for various number of topics
    :param dictionary: Dictionary
    :param corpus: Corpus
    :param texts: List of input texts
    :param limit: The maximum number of topics
    :param coherence: Coherence score to be used
    :param start: The minimum number of topics
    :param step: The step size for the number of topics to consider
    :param mallet_path: Path to mallet if you want to use it for the LDA model
    :param window_size: size of the window to be used for coherence measures using boolean sliding window as their probability estimator. For u_mass this doesn't matter. If None, the default window
                        sizes are used which are: c_v - 110, c_uci - 10, c_npmi - 10.
    :param args: Dictionary of parameters to send to LDA models (Gensim or Mallet)
    :return: list of LDA topic models and the list of corresponding coherence values
    """
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):
        if mallet_path is not None:
            model = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path,
                                                     corpus=corpus,
                                                     num_topics=num_topics,
                                                     id2word=dictionary,
                                                     **args)
        else:
            # Note: Gensim's built in LDA model uses a different inference algorithm than Mallet (which uses Gibbs sampling). Gibbs sampling is more precise,
            # but slower.  To get comparable results with the built in LDA model in Gensim, trying increasing the number of passes through the corpus
            # and the number of iterations from the default.
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=num_topics,
                                                    **args)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model,
                                         texts=texts,
                                         dictionary=dictionary,
                                         coherence=coherence,
                                         window_size=window_size)
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values


def evaluate_model_coherence(model_list, coherence_values, limit, start=2, step=3):
    """
    Evaluates the coherence of models and selects the best model from input list
    :param model_list: List of trained LDA models for different numbers of topics
    :param coherence_values: The corresponding Cv coherence values for each model
    :param limit: The maximum number of topics considered
    :param start: The minimum number of topics considered
    :param step: The step size between topics used during model construction
    :return: The best model, the optimal number of topics, the corresponding coherence score and its index in the input model list.
    """
    # Plot the coherence scores:
    x = range(start, limit, step)
    plt.figure(figsize=(20, 10))
    plt.plot(x, coherence_values)
    plt.xlabel("Number of topics")
    plt.ylabel('Cv coherence scores')
    plt.legend(['Coherence values'], loc='best')
    plt.show()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Number of topics: {0}  Cv coherence score: {1}".format(m, round(cv, 4)))

    # Select best model
    print("Selecting best model.  Note: this is based on maximizing coherence score.")
    print("If the score does not maximize, consider changing topic number range, or select the model that gave the highest coherence score before flattening out.")
    print("Examine the plot of number of topics vs coherence score to judge for yourself.")

    # Index of best coherence score
    best_index = coherence_values.index(max(coherence_values))
    best_model = model_list[best_index]

    print("The best model has {0} topics".format(x[best_index]))
    print("The topics are (shows 20 topics by default if possible):")
    # model_topics = best_model.show_topics(formatted=False)
    pprint(best_model.print_topics(num_words=10))

    return best_model, x[best_index], max(coherence_values), best_index


def format_topics_sentences(ldamodel, corpus, texts, topn=10, mallet=False):
    """
    Determine the dominant topic in a document (sentence).  Works with Mallet models.
    Use the function format_topics_sentences_gensim with gensim's built in LDA model.
    :param ldamodel: The trained LDA model
    :param corpus: The corpus used for the LDA model
    :param texts: The actual text to consider
    :param topn: The number of words to get per topic
    :param mallet: If a mallet model was used for LDA (the output format is a bit different)
    :return: DataFrame with the most dominant topic per document (in texts)
    """
    sent_topics_df = pd.DataFrame()
    if mallet:
        topics_dist_model = ldamodel[corpus]
    else:
        topics_dist_model = ldamodel.get_document_topics(corpus)
    # Get the main topic in each document
    for i, row in enumerate(topics_dist_model):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the dominant topic, percent contribution and keywords for each document
        for j, (topic_num, prob_topic) in enumerate(row):
            if j == 0:
                # This is the dominant topic
                wp = ldamodel.show_topic(topic_num, topn=topn)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prob_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['dominant_topic', 'percent_contribution', 'topic_keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


def format_topics_sentences_gensim(ldamodel, corpus, texts):
    """
    Determine the dominant topic in a document (sentence).  Works with gensim lda models.
    Use the function format_topics_sentences_mallet with mallet LDA models.
    :param ldamodel: The trained LDA model
    :param corpus: The corpus used for the LDA model
    :param texts: The actual text to consider
    :return: DataFrame with the most dominant topic per document (in texts)
    """
    sent_topics_df = pd.DataFrame()

    # Get the main topic in each document
    for i, row in enumerate(ldamodel.get_document_topics(corpus)):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, tup in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(tup[0])
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(tup[0]), round(tup[1], 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['dominant_topic', 'percent_contribution', 'topic_keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


def format_dominant_topics_df(df):
    """
    Formats the output of format_topics_sentences
    :param df: Output of format_topics_sentences
    :return: Formatted DataFrame
    """
    df_dominant_topic = df.reset_index()
    df_dominant_topic.columns = ['document_number', 'dominant_topic', 'topic_percent_contribution', 'keywords', 'text']

    return df_dominant_topic


def get_most_representative_document_per_topic(df):
    """
    Finds the most representative document by found topic.  Helps with interpreting some LDA topics.
    :param df: Input DataFrame.  Should be the output of format_topics_sentences
    :return: DataFrame with the most representative text per topic.
    """
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df.groupby('dominant_topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['percent_contribution'], ascending=[0]).head(1)], axis=0)

    # reset index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # format
    sent_topics_sorteddf_mallet.columns = ['topic_number', 'topic_percent_contribution', 'keywords', 'text']

    return sent_topics_sorteddf_mallet


def topic_distribution_across_docs(df):
    """
    Gets the distribution of topics across all documents
    :param df: DataFrame, should be the output of format_topics_sentences
    :return: DataFrame with the topic distribution across all documents
    """
    # Number of documents for each topic
    topic_counts = df['dominant_topic'].value_counts()
    topic_counts_df = pd.DataFrame(topic_counts)
    topic_counts_df.reset_index(inplace=True)
    topic_counts_df.columns = ['dominant_topic', 'number_of_documents']

    # Percentage of documents for each topic
    topic_contribution = topic_counts / topic_counts.sum()
    topic_contribution_df = pd.DataFrame(topic_contribution)
    topic_contribution_df.reset_index(inplace=True)
    topic_contribution_df.columns = ['dominant_topic', 'percentage_of_documents']

    # Topic number and keywords
    topic_num_keywords = df[['dominant_topic', 'topic_keywords']].drop_duplicates()

    # merge
    df_dominant_topics = topic_counts_df.merge(topic_contribution_df, on='dominant_topic')
    df_dominant_topics = df_dominant_topics.merge(topic_num_keywords, on='dominant_topic')

    return df_dominant_topics


def get_topics_by_document(ldamodel, corpus, mallet=False):
    """
    Gets the contribution of each topic for each document in the corpus
    :param ldamodel: The trained LDA model
    :param corpus: The corpus for the documents
    :param mallet: If a mallet LDA model was used or not (the output format is a bit different)
    :return: DataFrame with the contribution of each topic to the documents in the corpus
    """
    documents_topic_dists = {}

    if mallet:
        topic_dist_model = ldamodel[corpus]
    else:
        topic_dist_model = ldamodel.get_document_topics(corpus)

    for i, row in enumerate(topic_dist_model):
        row = sorted(row, key=lambda n: (n[1]), reverse=True)
        document_number = i
        topic_dict = {}
        for tup_ in row:
            topic_num = 'topic_{0}_contribution'.format(tup_[0])
            topic_dict[topic_num] = tup_[1]
        documents_topic_dists[document_number] = topic_dict

    document_topic_dists_df = pd.DataFrame.from_dict(documents_topic_dists, orient='index')
    document_topic_dists_df.reset_index(inplace=True)
    document_topic_dists_df.rename(index=str, columns={'index': 'document_number'}, inplace=True)

    if mallet:
        # TODO: fix this to work with output from Gensim LDA model.
        arrange_cols = ['document_number'] + ['topic_{0}_contribution'.format(x) for x in range(0, document_topic_dists_df.shape[1] - 1)]
        document_topic_dists_df = document_topic_dists_df[arrange_cols]

    return document_topic_dists_df


def extract_data_for_mallet_pyldaviz(ldamodel):
    """
    Function to format output of Mallet LDA models for use with pyLDAviz.
    Based largely on: http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276
    :param ldamodel: Trained Mallet LDA model
    :return: Dictionary with required info for pyLDAviz.  Use it by calling pyLDAviz.prepare(**data_to_viz)
    """

    # Get the path to the model statefile.
    # This looks like the easiest way to access alpha and beta parameters, unfortunately.
    state_file = ldamodel.fstate()

    # Get the alpha and beta parameters
    with gzip.open(state_file, 'r') as state:
        params = [x.decode('utf8').strip() for x in state.readlines()[1:3]]

    extracted_params = (list(params[0].split(":")[1].split(" ")), float(params[1].split(":")[1]))
    alpha_ = [float(x) for x in extracted_params[0][1:]]
    beta_ = extracted_params[1]

    # Get the remaining info in the statefile.
    # This is also contained in the dataset, but it's already here, so we might a well use it.
    df_state = pd.read_csv(state_file,
                           compression='gzip',
                           sep=' ',
                           skiprows=[1, 2])

    df_state['type'] = df_state['type'].astype(str)

    # Get the document lengths
    docs_ = df_state.groupby('#doc')['type'].count().reset_index(name='doc_length')

    # Get the vocabulary: the words in the documents and their frequencies.
    vocab_ = df_state['type'].value_counts().reset_index()
    vocab_.columns = ['type', 'term_freq']
    vocab_ = vocab_.sort_values(by='type', ascending=True)

    # Get the topic term matrix, phi:
    phi_df = df_state.groupby(['topic', 'type'])['type'].count().reset_index(name='token_count')
    phi_df = phi_df.sort_values(by='type', ascending=True)
    # Compute phi
    matrix_phi = phi_df.pivot(index='topic', columns='type', values='token_count').fillna(value=0)
    matrix_phi = matrix_phi.values + beta_

    phi = sklearn.preprocessing.normalize(matrix_phi, norm='l1', axis=1)

    # Generate the document-topic matrix theta:
    theta_df = df_state.groupby(['#doc', 'topic'])['topic'].count().reset_index(name='topic_count')
    # Compute theta:
    matrix_theta = theta_df.pivot(index='#doc', columns='topic', values='topic_count').fillna(value=0)
    matrix_theta = matrix_theta.values + alpha_

    theta = sklearn.preprocessing.normalize(matrix_theta, norm='l1', axis=1)

    # Format data for us in pyLDAviz

    data_to_viz = {'topic_term_dists': phi,
                   'doc_topic_dists': theta,
                   'doc_lengths': list(docs_['doc_length']),
                   'vocab': list(vocab_['type']),
                   'term_frequency': list(vocab_['term_freq'])}

    return data_to_viz


def remove_ascii_words(df, text_col, replace_word):
    """
    Function to remove non ascii text from a text column in a dataframe.  The text column is modified in place.
    :param df: The dataframe with the text column you want to clean of non ascii text
    :param text_col: The name of the text column to consider
    :param replace_word: The word you want to replace non ascii characters with.  It is often helpful to use a nonsense word here as the presence of non standard characters might be important for
                         building predictive models.
    :return: The list of the non ascii words found if you want to inspect them.
    """
    # Initialize list to store discovered non ascii words
    non_ascii_words = []

    for i in range(len(df)):
        for word in df.loc[i, text_col].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                df.loc[i, text_col] = df.loc[i, text_col].replace(word, replace_word)

    return non_ascii_words


def w2v_preprocessing(df, text_col):
    """
    Function to preprocess a text column for word-2-vec modeling
    :param df: Dataframe with text you want to process
    :param text_col: The name of the text column to consider
    :return: Nothing. All manipulations are applied in place.
    """
    df[text_col] = df.text.str.lower()
    # Split out text into individual sentences
    df['document_sentences'] = df.text.str.split('.')

    # Tokenize and remove unwanted characters
    # TODO: Make unwanted character removal optional
    df['tokens'] = list(map(lambda sentences: list(map(sent_to_words, sentences)), df.document_sentences))
    # remove empty lists
    df['tokens'] = list(map(lambda sentences: list(filter(lambda lst: lst, sentences)), df.tokens))


def find_combos(search_str, search_term, before=False):
    """
    Function to find and produce combinations of words and numbers.  For example, you could turn type 2 into type_2.  This helps with producing relevant tokens for topic modeling that might otherwise
    be left out. With the above example, if you aren't careful, you might turn type 2 into just type.  Doing this explicitly when you know that there are some relevant concepts can be beneficial for
    your model.
    :param search_str: The string you want to search. Could be a full document, or a sentence within a document, etc.
    :param search_term: The word you want to look for. e.g. "type" like in the above example to make tokens like type_2
    :param before: Boolean. Set true if you want to search for numbers occurring before the search term. e.g. 24 years with before True to make 24_years
    :return: The modified input string and the original string. e.g. an input "study of type 2 diabetes" would return "study of type_2 diabetes as well as the original "study of type 2 diabetes". If
             no changes are made to the input string, the return_string variable will be the original input unchanged and the original_string variable will be None. You can use this to detect strings
             that have actually been modified.
    """
    if before:
        list_ = re.findall('[(\d+\s+)]*(?={0})'.format(search_term), search_str)
        try:
            values = list_[0].strip(' ')
            original = values + ' ' + search_term
            values = values.replace(' ', '_')
            string_return = values + '_' + search_term
        except IndexError:
            string_return = search_str
            original = None
    else:
        list_ = re.findall('(?<={0})[(\s+\d+)]*'.format(search_term), search_str)
        try:
            values = list_[0].strip(' ')
            original = search_term + ' ' + values
            values = values.replace(' ', '_')
            string_return = search_term + '_' + values
        except IndexError:
            string_return = search_str
            original = None

    return string_return, original


def string_sub(search_str, search_term, before=False):
    """
    Function to run the find_combos function. You can run this with a pandas DataFrame as an apply on a text column to run find_combos on all rows.
    :param search_str: The string you want to search. Could be a full document, or a sentence within a document, etc.
    :param search_term: The word you want to look for. e.g. "type" like in the above example to make tokens like type_2
    :param before: Boolean. Set true if you want to search for numbers occurring before the search term. e.g. 24 years with before True to make 24_years
    :return: The input string with the replaced terms. e.g. an input "study of type 2 diabetes" would return "study of type_2 diabetes
    """
    string_return, original = find_combos(search_str=search_str, search_term=search_term, before=before)
    if original is not None:
        replaced_str = search_str.replace(original, string_return)
    else:
        replaced_str = search_str

    return replaced_str

from __future__ import division

import os
import gzip
import re
import pandas as pd
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import pyLDAvis
import pyLDAvis.gensim

import nltk

from langdetect import detect

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
            print(lang_name)
            words_to_return += nltk.corpus.stopwords.words(lang_name)
            print(len(words_to_return))
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


def sent_to_words(data_list):
    """
    Tokenize words and remove punctuation.
    :param data_list: List containing sentences to process
    :return: List of tokenized sentences
    """
    # Tokenize words and remove punctuation
    for sentence in tqdm(data_list):
        yield (gensim.utils.simple_preprocess(unicode(sentence), deacc=True))


def remove_stopwords(texts, stop_words):
    """
    Removes stopwords from input text
    :param texts: Text to process
    :param stop_words: List of stop words to filter.
    :return: List of Lists.  Each document becomes a list of processed words without stopwords
    """
    # return [[word for word in simple_preprocess(unicode(doc)) if word not in simple_preprocess(unicode(stop_words))] for doc in tqdm(texts)]
    return [[word for word in doc if word not in stop_words] for doc in tqdm(texts)]


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


def lda_preprocess(df, text_col, lang_list, min_count=5, threshold=100, trigrams=False, custom_remove=None):
    """
    Applies the functions above to preprocess text data for
    :param df: A DataFrame which contains the text you want to process
    :param text_col: The column name in df which contains the text
    :param lang_list: List of languages you want to consider. See https://pypi.org/project/langdetect/ for supported languages
    :param min_count: min_count for bigram model.  The higher this is, the harder it is to make bigrams
    :param threshold: threshold for bigram model.  The higher this is, the harder it is to make bigrams
    :param trigrams: True if you want to lemmatize on trigrams, default is False
    :param custom_remove: List of additional words to remove (treat as stopwords).
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
    print('Generating stopword list for specified languages: {0}'.format(df_lang['language'].unique().tolist()))
    stop_words = get_stopwords(lang_list)
    if custom_remove:
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
        data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ABV'])
    else:
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ABV'])
    print('End lemmatization')

    print('Begin creating dictionary')
    id2word = create_dictionary(data_lemmatized)
    print('End creating dictionary')

    print('Being creating corpus')
    corpus = create_corpus(id2word, data_lemmatized)
    print('End creating corpus')

    print('End LDA preprocessing')

    return id2word, corpus, data_lemmatized


def compute_coherence_values(dictionary, corpus, texts, limit, coherence='c_v', start=2, step=3, mallet_path=None, args={}):
    """
    Compute the Cv coherence for various number of topics
    :param dictionary: Dictionary
    :param corpus: Corpus
    :param texts: List of input texts
    :param limit: The maximum number of topics
    :param start: The minimum number of topics
    :param step: The step size for the number of topics to consider
    :param mallet_path: Path to mallet if you want to use it for the LDA model
    :param args: Dictionary of parameters to send to LDA models (Gensim or Mallet)
    :return: list of LDA topic models and the list of corresponding coherence values
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        if mallet_path:
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
                                         coherence=coherence)
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
    print("The topics are:")
    # model_topics = best_model.show_topics(formatted=False)
    pprint(best_model.print_topics(num_words=10))

    return best_model, x[best_index], max(coherence_values), best_index


def format_topics_sentences(ldamodel, corpus, texts, mallet=False):
    """
    Determine the dominant topic in a document (sentence).  Works with Mallet models.  
    Use the function format_topics_sentences_gensim with gensim's built in LDA model.
    :param ldamodel: The trained LDA model
    :param corpus: The corpus used for the LDA model
    :param texts: The actual text to consider
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
                wp = ldamodel.show_topic(topic_num)
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
    topic_contribution = topic_counts/topic_counts.sum()
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
                           skiprows=[1,2])
    
    df_state['type'] = df_state['type'].astype(str)
    
    # Get the document lengths
    docs_ = df_state.groupby('#doc')['type'].count().reset_index(name ='doc_length')
    
    # Get the vocabulary: the words in the documents and their frequencies.
    vocab_ = df_state['type'].value_counts().reset_index()
    vocab_.columns = ['type', 'term_freq']
    vocab_ = vocab_.sort_values(by='type', ascending=True)
    
    # Get the topic term matrix, phi:
    phi_df = df_state.groupby(['topic', 'type'])['type'].count().reset_index(name ='token_count')
    phi_df = phi_df.sort_values(by='type', ascending=True)
    # Compute phi
    matrix_phi = phi_df.pivot(index='topic', columns='type', values='token_count').fillna(value=0)
    matrix_phi = matrix_phi.values + beta_
    
    phi = sklearn.preprocessing.normalize(matrix_phi, norm='l1', axis=1)
    
    # Generate the document-topic matrix theta:
    theta_df = df_state.groupby(['#doc', 'topic'])['topic'].count().reset_index(name ='topic_count')
    # Compute theta:
    matrix_theta = theta_df.pivot(index='#doc', columns='topic', values='topic_count')
    matrix_theta = matrix_theta.values + alpha_

    theta = sklearn.preprocessing.normalize(matrix_theta, norm='l1', axis=1)
    
    # Format data for us in pyLDAviz 
    
    data_to_viz = {'topic_term_dists': phi, 
                   'doc_topic_dists': theta,
                   'doc_lengths': list(docs_['doc_length']),
                   'vocab': list(vocab_['type']),
                   'term_frequency': list(vocab_['term_freq'])}
    
    return data_to_viz


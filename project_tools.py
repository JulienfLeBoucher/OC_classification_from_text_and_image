import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import re
import cv2 as cv

# Define collection of words/tokens
stopwords = nltk.corpus.stopwords.words('english')
correct_words = set(nltk.corpus.words.words())
# Define lemmatizer and stemmer
lemmatizer = nltk.stem.WordNetLemmatizer()
porter = nltk.stem.PorterStemmer()
lancaster = nltk.stem.LancasterStemmer()


#### Note (to self) ####
# I chose to constitute a dataframe with columns containing different list 
# of tokens (from distinct pre-processing methods.)
# I saw that some people sometime join the tokens into a new string.
# This can be easier to handle compatibility with the Sklearn library.
# May be I'll use this in later projects if it is found to be more convenient
# using a rejoin parameter in my tokenize functions.

def tokenize_product_name(product_name: str, rejoin: bool=False)-> list:
    # Define tokens we want to filter in the word_tokenize() result.
    useless_tokens = ["'s", "''"]
    undesired_tokens = [*useless_tokens,
                        *string.punctuation,
                        *stopwords,]
    
    # Tokenize and lower case.
    tokens = nltk.word_tokenize(product_name)
    tokens = [token.lower() for token in tokens]
    # Filter undesired tokens
    tokens = [token
              for token in tokens
              if token not in undesired_tokens]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if rejoin:
        tokens = ' '.join(tokens)

    return tokens


def tokenize_description(description: str, rejoin: bool=False)-> list:
    # Define tokens we want to filter because they were found not relevant
    # in helping classifying the products.
    useless_tokens = ["'s", "''"]
    undesired_tokens = [*useless_tokens,
                        *string.punctuation,
                        *stopwords,]
    
    # If no space between basic punctuation and the following capital letter,
    # Add one to ensure separation in tokenization.
    regex = '([.?,!])([A-Z])'
    matches = re.finditer(regex, description)
    description = re.sub(regex, r"\1 \2", description)
    
    # Tokenize and lower case.
    tokens = nltk.word_tokenize(description)
    tokens = [token.lower() for token in tokens]
    # Filter undesired tokens
    tokens = [token
              for token in tokens
              if token not in undesired_tokens]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if rejoin:
        tokens = ' '.join(tokens)

    return tokens


def tokenize_description_v2(
    description: str,
    rare_tokens: list=[],
    filter_numeric=False,
    rejoin: bool=False,
)-> list:
    """ Return the tokenized description following those steps:
    
    1) Address the problem where the first word of a supposed-sentence
       adjoins the final dot of the previous one such as in:
       'First sentence.Second one'
       
    2) Tokenize and ensure lower case.
    
    3) Lemmatize.
    
    4) Filter noisy tokens with regard to the expected classification
       such as : 
        - punctuation;
        - stopwords;
        - rare tokens find in the standard tokenization;
        - tokens very frequently present in several categories.
        
        with an option to remove numeric tokens.
        
    if rejoin is set to True, then return a string.
    """
    ############ Define tokens to be dropped ##########
    # Tokens we want to filter because do not help in classifying
    # products. At least present massively in the top 50 of 4 categories 
    # out of the 7.
    useless_tokens = ['product', 'price', 'free', 'r', 'buy',
                      'color', 'day', 'genuine', 'cash', 'delivery',
                      'shipping', 'feature', 'pack', 'replacement',
                      'specification', '1', 'set', 'guarantee', '30',
                      'flipkart.com',

                      ]
    
    # Concatenate all undesired tokens
    undesired_tokens = ["'s", "''",
                        *useless_tokens,
                        *string.punctuation,
                        *rare_tokens,
                        *stopwords,
                        ]
    
    ##### STEP 1 #####
    # If no space between basic punctuation and the following capital letter,
    # Add one to ensure separation in tokenization.
    regex = '([.?,!])([A-Z])'
    matches = re.finditer(regex, description)
    description = re.sub(regex, r"\1 \2", description)
    
    ##### STEP 2 #####
    # Tokenize and lower case.
    tokens = nltk.word_tokenize(description)
    tokens = [token.lower() for token in tokens]
    
    ##### STEP 3 #####
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    ##### STEP 4 #####
    # Filter undesired tokens
    tokens = [token
              for token in tokens
              if token not in undesired_tokens]
    
    if filter_numeric:
        tokens = [token for token in tokens if not token.isnumeric()]
    
    ##### Optional #####
    if rejoin:
        tokens = ' '.join(tokens)

    return tokens


def tokenize_description_v3(
    description: str,
    rare_tokens: list=[],
    filter_numeric=False,
    rejoin: bool=False,
)-> list:
    """ Return the tokenized description following those steps:
    
    1) Address the problem where the first word of a supposed-sentence
       adjoins the final dot of the previous one such as in:
       'First sentence.Second one'
       
    2) Tokenize and ensure lower case.
    
    3) Lemmatize.
    
    4) Filter noisy tokens with regard to the expected classification
       such as : 
        - punctuation;
        - stopwords;
        - rare tokens find in the standard tokenization;
        - MORE tokens very frequently present in several categories than
        v2.
        
        with an option to remove numeric tokens.
        
    if rejoin is set to True, then return a string.
    """
    ############ Define tokens to be dropped ##########
    # Tokens we want to filter because do not help in classifying
    # products. At least present massively in the top 100 tokens in 
    # 3 categories out of the 7.
    useless_tokens = [
       '1', 'specification', 'genuine', 'buy', 'general', 'color', 'free',
       'shipping', 'box', 'delivery', 'feature', 'type', 'cash', 'product',
       'day', 'online', 'set', 'material', 'r', 'brand', 'price',
       'flipkart.com', 'made', 'key', 'guarantee', 'model', '30', 'number',
       'replacement', 'size', 'pack', '2', 'inch', 'black', 'warranty', 'best',
       'print', 'quality', 'sale', 'package', 'wash', 'multicolor', 'gift',
       'yes', 'code', 'care', 'cm', 'ideal', 'blue', '4', 'pink', 'style',
       'dimension', 'design',
                      ]
    
    # Concatenate all undesired tokens
    undesired_tokens = ["'s", "''",
                        *useless_tokens,
                        *string.punctuation,
                        *rare_tokens,
                        *stopwords,
                        ]
    
    ##### STEP 1 #####
    # If no space between basic punctuation and the following capital letter,
    # Add one to ensure separation in tokenization.
    regex = '([.?,!])([A-Z])'
    matches = re.finditer(regex, description)
    description = re.sub(regex, r"\1 \2", description)
    
    ##### STEP 2 #####
    # Tokenize and lower case.
    tokens = nltk.word_tokenize(description)
    tokens = [token.lower() for token in tokens]
    
    ##### STEP 3 #####
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    ##### STEP 4 #####
    # Filter undesired tokens
    tokens = [token
              for token in tokens
              if token not in undesired_tokens]
    
    if filter_numeric:
        tokens = [token for token in tokens if not token.isnumeric()]
    
    ##### Optional #####
    if rejoin:
        tokens = ' '.join(tokens)

    return tokens


def make_tokens_count(df, colname_with_token):
    # concatenate all tokens from names in a Series. 
    # Afterwards, count each occurence in it in 
    # a df with 2 columns (token, count)
    return pd.DataFrame(

        pd.Series(     
            df[colname_with_token]
            .sum()
        ).value_counts()

    ).reset_index(names='token')


def find_top_tokens(
    df, colname_with_token,
    n_top=50,
    n_display=30,
    display=False,
    ax=None,
):
    """ return the n_top top-tokens in occurrences.

    If display=True, display a bar diagram of the n_display top
    tokens occurrences. """
    counts = make_tokens_count(df, colname_with_token).set_index('token')
    
    if display:
        fig, ax = plt.subplots(figsize=(n_display/5, 5))
        counts.head(n_display).plot(
            ax=ax,
            kind='bar', 
            title=f"top-{n_display} tokens",
        )
        # plt.legend(False)
        plt.show()
    
    if ax is not None:
        counts.head(n_display).plot(
            ax=ax,
            kind='bar', 
        )
    return counts.index[:n_top]


#### LDA and NMF
def plot_top_words(
    model, feature_names, n_top_words, title, n_cols=4, n_rows=2):
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*7, n_rows*7),
        sharex=True
    )
    axes = axes.flatten()
    # plot top words of each topic with their weight
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    # Hide unused axes.
    for n in range(n_rows * n_cols):
        if n > topic_idx:
            axes[n].set_visible(False)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
    
def get_topic_label(fit_model, X):
    """ label each individual wth its most probable latent topic
        from a fit LDA or NMF object """
    return np.argmax(fit_model.transform(X), axis=1)


#### Word2Vec ####
def get_word_vector(word, model):
    """ Get the vector associated to 'word' by a
        word2vec model (gensim).
        
        If the word is not in the vocabulary,
        the null vector is returned. """
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros((model.vector_size,))


def get_sentence_vector(sentence, model, normalize=True):
    """ return the sum of all vectors in a sentence.
        Normalize the vector if 'normalize' is True. """
    if normalize:
        sum_vector = sum(get_word_vector(w, model) for w in sentence)
        norm = np.linalg.norm(sentence_vector, 2)
        return sum_vector / norm
    else:
        return sum(get_word_vector(w, model) for w in sentence)
    
    
def resize_based_on_lowest_dimension(img, n_pixels=200):
    """ Resize an image preserving its shape while mapping its short 
        side to 'n_pixels'."""
    # Find the ratio between the desired image and the image fed-in.
    ratio = (n_pixels / min(img.shape[:2]))
    # Store the short side rank
    low_dim = np.argmin(img.shape[:2])
    # Resize while ensuring there is no rounding problems 
    # on the short side which could end up being 199 if multiply doing
    # the maths.
    if low_dim == 0:
        width = int(img.shape[1] * ratio)
        height = int(n_pixels)
    else:
        width = int(n_pixels)
        height = int(img.shape[0] * ratio)
        
    new_dim = (width, height)
    return cv.resize(img, new_dim, interpolation = cv.INTER_AREA)


def create_image_vector_from_sift_descriptors(img_descriptors, fit_kmeans):
    # Compute labels of each descriptor in the image.
    # float64 is needed for buffer compatibility.
    labels = fit_kmeans.predict(np.float64(img_descriptors))
    # Instantiate a null vector of length n_clusters and fill it
    # to buid a normalized histogram.
    img_vect = np.zeros(fit_kmeans.cluster_centers_.shape[0])
    for unique, count in zip(*np.unique(labels, return_counts=True)):
        img_vect[unique] = count / len(img_descriptors)
    return img_vect


def create_image_vector_from_descriptors(img_descriptors, fit_kmeans):
    # Compute labels of each descriptor in the image.
    # float64 is needed for buffer compatibility.
    labels = fit_kmeans.predict(np.float64(img_descriptors))
    # Instantiate a null vector of length n_clusters and fill it
    # to buid a normalized histogram.
    img_vect = np.zeros(fit_kmeans.cluster_centers_.shape[0])
    for unique, count in zip(*np.unique(labels, return_counts=True)):
        img_vect[unique] = count / len(img_descriptors)
    return img_vect


def compute_labels_occurrences_in_all_images(descriptors_list, fit_kmeans):
    # Instantiate the null vector (count = 0 for all visual words)
    # with the right shape
    labels_occurrences = np.zeros(fit_kmeans.cluster_centers_.shape[0])
    # Goes through each image, find the visual words, and update 
    # the count of its presence in the entire list.
    for img_descriptors in descriptors_list:
        for unique in np.unique(
            fit_kmeans.predict(np.float64(img_descriptors))
        ):
            labels_occurrences[unique] += 1
    return labels_occurrences
    
    
def create_tfidf_image_vector_from_sift_descriptors(
        img_descriptors,
        corpus_labels_occurrences,
        n_images,
        fit_kmeans,
    ):
    """ Return the normalized tfidf image vector based on
    its SIFT features mapped to the visual words vocabulary."""
    # Compute labels of each descriptor in the image.
    # float64 is needed for buffer compatibility.
    labels = fit_kmeans.predict(np.float64(img_descriptors))
    # Instantiate a null vector of length n_clusters (length of the
    # visual words vocabulary) and fill it with the count, normalized 
    # by the inverse of the document frequency 'idf'.
    img_vect = np.zeros(fit_kmeans.cluster_centers_.shape[0])
    for unique, count in zip(*np.unique(labels, return_counts=True)):
        idf = np.log(n_images/ corpus_labels_occurrences[unique])
        img_vect[unique] = count * idf
    return img_vect / sum(img_vect)
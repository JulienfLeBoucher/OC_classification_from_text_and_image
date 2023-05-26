import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import re

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
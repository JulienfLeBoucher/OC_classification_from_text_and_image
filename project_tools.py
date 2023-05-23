import pandas as pd
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

def final_tokenize_description(description: str, rejoin: bool=False)-> list:
    # Tokens we want to filter because they were found not relevant
    # in helping to classify the products. 
    # (At least present massively in 3 categories or more out of the 7)
    useless_tokens = ["'s", "''", 'product', 'price', 'free', 'r', 'buy',
                      'color', 'day', 'genuine', 'cash', 'delivery',
                      'shipping', 'feature', 'pack', 'replacement',
                      'specification', '1', 'set', 'guarantee', '30',
                      'flipkart.com',
                    ### AND SOME WHICH COULD BE ADDED TOO  
                    #   'material', 'cm', 'type', 'online', 'sale', 'box',
                    #   'design', 'package',
                      ]
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
    df, colname_with_token, n_top=50,
    n_display=30, display=False
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
    return counts.index[:n_top]


  
import requests
import pandas as pd
import config
from numpy import NaN

########################################################
# To use this script, sign up on rapidapi.com and use your
# X_RapidAPI_Key in the headers.
#
# It was made to extract the first `n_items` products in relation with 
# the 'product_of_interest' having the `features` features.

################ Parameters ############################
product_of_interest = "champagne"
# Number of items fully provided to extract
n_items = 10
# Name of the features to be extracted
features = ['foodId', 'label', 'category', 'foodContentsLabel', 'image']
# Number of pages of 20 products to be retrieved
n_pages = 5
# For saving results
csv_filename = f'./{n_items}_{product_of_interest}_products.csv'
########################################################

def extract_product_features(product_info, features):
    """ Extract the provided features in `features` and fill 
        non-provided features with NaN.
    
        Return a Dataframe where each line is a feature from `features`.
    
        - `product_info` is a dictionary in which keys are supposed to match
           with `features`.
        - `features` a list of desired features."""
    # Get the provided features.
    my_dict = {
        ft: val for ft, val in product_info.items()
        if ft in features
    }
    # Add missing features filled with the NaN value.    
    my_dict.update({ft: NaN for ft in features if ft not in my_dict.keys()})
    return pd.DataFrame.from_dict(my_dict, orient='index')


############### API REQUESTS ############################
url = "https://edamam-food-and-grocery-database.p.rapidapi.com/api/food-database/v2/parser"

headers = {
	"X-RapidAPI-Key": config.X_RapidAPI_Key,
	"X-RapidAPI-Host": "edamam-food-and-grocery-database.p.rapidapi.com"
}

querystr = {"ingr":product_of_interest}

data =[]
for k in range(n_pages):
    response = requests.request("GET", url, headers=headers, params=querystr)
    
    if response.status_code == 200:
        # Retrieve the relevant information
        data += response.json()['hints']
        # Get next page url
        url = response.json()['_links']['next']['href']
            
    elif response.status_code == 400:
        print('400 Bad Request')
        break

######################## KEEPING THE RELEVANT INFORMATION ############
# Extract relevant information in `items`.
# This contains 'foodId', 'label', 'category', 'foodContentsLabel', 'image'.
# It might need to be changed if one wants to get extra features.
items = pd.DataFrame(data).food

dfs = []
# Extract a df per product
for item in items:
    dfs.append(extract_product_features(item, features))

# Concat and drop partially provided products.    
r_data = (
    pd.concat(dfs, axis=1)
    .T
    .dropna(axis=0)
    .drop_duplicates()
)

# Write results to a csvfile if a sufficient number of products were found.
if r_data.shape[0] >= n_items:
    r_data.head(10).to_csv(csv_filename, index=False)
else:
    print('Not enough well-provided products were retrieved from the API\n'
          '--> Increase the number via the `n_pages` parameter')
# Products classification based on a picture and a text description.

**Main goals** :
- Assess the feasibility of clustering products sold on an indian e-platform thanks to the description of the product and/or its image.
- Provide an image-based classification model that uses data augmentation.
- Develop a script to extract data of certain products through an API.

In order to do that I develop several notebooks.

## Natural Language Processing

- The [first notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_classification_from_text_and_image/blob/main/NLP.ipynb) is about discovering text pre-processing and basic vectorization (bags of words and tf-idf). It is applied on the name of the products and first assessments of the feasibility of classifying are made.

- The [second notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_classification_from_text_and_image/blob/main/NLP2.ipynb) is about deepening and varying the pre-processing steps and apply it on the full text description concatenated with the product name. I refined my evaluation strategy and used a mix of dimensionality reduction techniques and clustering techniques such as Kmeans, Gaussian Mixture, Latent Dirichlet Allocation and Non-negative Matrix Factorization.

- The [third notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_classification_from_text_and_image/blob/main/NLP3.ipynb) consists in improving the vectorization step using word and sentence embeddings
outputs given by Word2Vec, sBERT and USE models.

## Image Processing
- In the [fourth notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_classification_from_text_and_image/blob/main/SIFT.ipynb), I discovered the OpenCV library and extracted image features obtained with the SIFT method. I then built a reduced 'feature vocabulary' using a clustering technique in order to characterize the content of each image with a relative low dimension vector. There, I assessed the classification feasibility.
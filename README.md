# Home Depot Product Search Relevance

A notebook-based machine learning project that predicts how relevant a Home Depot product is to a search query. It uses search terms, product titles, and product descriptions to compare text representations and regression models on the Home Depot Product Search Relevance dataset from Kaggle.

## Project files

| File | Description |
| --- | --- |
| [Home depot product search relevance.ipynb](Home%20depot%20product%20search%20relevance.ipynb) | Data preparation, model training, evaluation, and visualizations. |
| [Home depot product search relevance - outputs.html](Home%20depot%20product%20search%20relevance%20-%20outputs.html) | Exported notebook outputs; download and open in a browser to view locally. |
| [report.pdf](report.pdf) | Project report. |

## Approach

- **Data augmentation:** samples 4% of the training data and creates English, Spanish, French, and German variants of search terms and product titles. Product descriptions remain unchanged.
- **Character-level Siamese LSTM:** shares an encoder between queries and product text, then combines their representations and absolute difference to predict relevance. Inputs are truncated or padded to 40 query characters and 400 product characters.
- **TF-IDF baseline:** uses up to 5,000 features from product titles and search terms with Ridge regression.
- **Word2Vec experiment:** tokenizes the text and trains 100-dimensional word embeddings.
- **Sentence-BERT features:** uses `sentence-transformers/all-MiniLM-L6-v2` embeddings, combining query and product vectors with their absolute difference and elementwise product. Standardized features feed an MLP regressor.

Evaluation includes mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), training curves, and prediction plots.

## Run in Google Colab

The notebook is written for Google Colab and uses Google Drive for dataset storage. Dataset files are not included in this repository.

1. Obtain the Home Depot Product Search Relevance competition data from Kaggle.
2. Place these archives in the root of your Google Drive (`MyDrive`), matching the notebook's setup cell:

   ```text
   attributes.csv.zip
   product_descriptions.csv.zip
   sample_submission.csv.zip
   test.csv.zip
   train.csv.zip
   home-depot-product-search-relevance.zip
   ```

   The CSV archives should extract to their corresponding CSV filenames. The notebook reads `train.csv`, `test.csv`, `product_descriptions.csv`, and `attributes.csv`; its setup also copies and extracts the sample submission and full competition archives.

3. Upload or open `Home depot product search relevance.ipynb` in Colab. A GPU runtime can help with neural model training and embedding generation.
4. Run the Drive mount cell and authorize access. Adjust the `/content/drive/MyDrive/` paths if your archives are stored elsewhere.
5. Run the remaining cells in order. The setup cell deletes and recreates its dataset directories under `/content`, then copies and extracts the archives.

The notebook installs `deep-translator`, `googletrans==4.0.0-rc1`, `gensim`, and `sentence-transformers` in individual cells. It also uses NumPy, pandas, Matplotlib, scikit-learn, TensorFlow/Keras, NLTK, and tqdm. If these are missing from your runtime, install them before running the relevant cells:

```python
%pip install numpy pandas matplotlib scikit-learn tensorflow nltk tqdm
```

Internet access is needed for package installation, translation, the NLTK resource download, and the first Sentence-BERT model download. Translation runs row by row and can take time; failed translations silently retain the original text.

## Run locally

Use a Python environment with Jupyter and the dependencies above. Replace the Google Drive mount and Colab shell setup cells with local dataset preparation, then update the four `pd.read_csv` paths. The existing setup uses Linux shell commands and `/content` paths, so it requires adaptation for a local Windows environment.

## Experiment notes

- The section labeled **Word-level Siamese LSTM** currently rebuilds the character encoder and trains on the same character inputs. The trained Word2Vec embeddings are not connected to that model.
- Augmentation and some feature fitting happen before validation splitting. Related or duplicate examples can cross splits, and vocabulary fitting can include validation text, so the recorded metrics should be treated as exploratory results.
- The notebook does not pin a complete environment or seed every source of randomness; results may vary between runs.
- The test data is loaded, but the notebook does not generate a submission file or save trained models.

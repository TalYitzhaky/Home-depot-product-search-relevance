# Home Depot Product Search Relevance

A notebook-based machine learning project that predicts how relevant a Home Depot product is to a search query. It uses search terms, product titles, and product descriptions to compare text representations and regression models on the Home Depot Product Search Relevance dataset from Kaggle.

## Project files

| File | Description |
| --- | --- |
| [Home depot product search relevance.ipynb](Home%20depot%20product%20search%20relevance.ipynb) | Data preparation, model training, evaluation, and visualizations. |
| [Home depot product search relevance - outputs.html](Home%20depot%20product%20search%20relevance%20-%20outputs.html) | Historical notebook outputs from before the leakage fix; download and open in a browser to view locally. |
| [report.pdf](report.pdf) | Historical project report from before the leakage fix. |

## Approach

- **Data augmentation:** splits original examples into 80% training and 20% validation (`random_state=42`), then samples 4% of only the training partition and creates English, Spanish, French, and German variants of search terms and product titles. Product descriptions remain unchanged.
- **Character-level Siamese LSTM:** shares an encoder between queries and product text, then combines their representations and absolute difference to predict relevance. Inputs are truncated or padded to 40 query characters and 400 product characters.
- **TF-IDF baseline:** uses up to 5,000 features from product titles and search terms with Ridge regression.
- **Word-level Siamese LSTM:** tokenizes text and trains 100-dimensional Word2Vec embeddings on training tokens only. A shared LSTM uses those embeddings, fine-tunes them during training, and processes sequences capped at 40 query words and 400 product words. Padding is masked; unseen or rare words map to an unknown-word ID.
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
5. Restart the runtime and run all cells in order to avoid reusing variables from earlier experiments. The setup cell deletes and recreates its dataset directories under `/content`, then copies and extracts the archives.

The notebook installs `deep-translator`, `googletrans==4.0.0-rc1`, `gensim`, and `sentence-transformers` in individual cells. It also uses NumPy, pandas, Matplotlib, scikit-learn, TensorFlow/Keras, NLTK, and tqdm. If these are missing from your runtime, install them before running the relevant cells:

```python
%pip install numpy pandas matplotlib scikit-learn tensorflow nltk tqdm
```

Internet access is needed for package installation, translation, the NLTK resource download, and the first Sentence-BERT model download. Translation runs row by row and can take time; failed translations silently retain the original text.

## Run locally

Use a Python environment with Jupyter and the dependencies above. Replace the Google Drive mount and Colab shell setup cells with local dataset preparation, then update the four `pd.read_csv` paths. The existing setup uses Linux shell commands and `/content` paths, so it requires adaptation for a local Windows environment.

## Experiment notes

- All models use the same split of original example IDs. Augmented variants stay in training; validation examples are not augmented. Different examples involving the same product may occur in both partitions.
- Character vocabulary, TF-IDF, Word2Vec, and feature scaling are fitted only on training data. Validation uses those fitted transformations and the fixed pretrained Sentence-BERT encoder.
- Both LSTMs use explicit validation data and early stopping on validation loss (patience 3, best weights restored, at most 10 epochs). The MLP uses an explicit epoch loop with validation R² (patience 10, tolerance 0.0001, at most 200 epochs), restores the best model, and creates no internal validation split.
- Validation guides early stopping; its metrics are not an independent final test score. Notebook outputs have been cleared and must be regenerated. The existing HTML export and PDF report retain historical results from before the leakage fix.
- The notebook does not pin a complete environment or seed every source of randomness; results may vary between runs.
- The test data is loaded, but the notebook does not generate a submission file or save trained models.

## Tests

[tests/test_data_leakage.py](tests/test_data_leakage.py) contains offline regression tests for the train/validation leakage fix. They verify that augmentation and fitted preprocessing use training data only, validation data remains untouched, and early stopping uses the explicit validation set.

The tests intentionally execute actual notebook cells. Heavy ML dependencies and translation are replaced with lightweight offline substitutes, so running the tests requires no original dataset, pretrained model downloads, or external APIs. This is targeted regression coverage for the leakage fix, not a comprehensive project test suite.

With NumPy, pandas, scikit-learn, and nbformat installed, run:

```sh
python -m unittest discover -s tests -v
```

The checks also cover feature alignment, best-model restoration, and notebook structure. Named cell references are centralized in `NOTEBOOK_CELLS`, with source checks that report unexpected layout changes. Moving or rewriting those cells may require updating the references; the tests do not retrain the full notebook.

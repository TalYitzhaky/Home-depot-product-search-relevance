"""Offline regression tests for the Home Depot Product Search Relevance pipeline.

These tests verify that the train/validation split occurs before augmentation
and feature fitting, preventing validation-data leakage. This repository is
notebook-based, so the tests intentionally execute actual notebook cells rather
than duplicate their preprocessing logic.

Translation, TensorFlow, Word2Vec, and Sentence-BERT are replaced with lightweight
offline substitutes; no datasets, pretrained models, or external APIs are needed.
Python, pandas, NumPy, scikit-learn, and nbformat must already be installed.
These are regression tests for the leakage fix, not a complete project test suite.
"""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / 'Home depot product search relevance.ipynb'
NOTEBOOK_DOCUMENT = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))

# Zero-based notebook positions and identifying source fragments. Keep this map
# in sync when cells move; markers turn stale positions into actionable errors.
NOTEBOOK_CELLS = {
    'safe_translate': (12, 'def safe_translate('),
    'augment_row': (13, 'def augment_row('),
    'split_originals': (14, 'train_original, validation_original = train_test_split('),
    'generate_augmentations': (15, 'augmented_data = []'),
    'augmentation_frame': (16, 'augmented_sample = ('),
    'combine_training': (17, 'train_augmented = pd.concat('),
    'partition_counts': (18, 'print("Original labeled examples:"'),
    'character_features': (20, 'def prepare_text('),
    'tfidf_features': (30, 'tfidf_vectorizer = TfidfVectorizer('),
    'tfidf_partitions': (32, 'X_train, X_val = tfidf_train, tfidf_val'),
    'ridge_fit': (33, 'baseline_model.fit('),
    'ridge_metrics': (34, 'y_train_pred = baseline_model.predict('),
    'word2vec_fit': (38, 'w2v_model = Word2Vec('),
    'word_features': (40, 'def build_word_encoder('),
    'word_model': (41, 'word_encoder = build_word_encoder('),
    'first_lstm_fit': (25, 'history = model1.fit('),
    'second_lstm_fit': (43, 'history = model2.fit('),
    'sentence_features': (47, 'def sentence_features('),
    'sentence_partitions': (48, 'X_train, X_val = sbert_train, sbert_val'),
    'scale_features': (49, 'scaler = StandardScaler('),
    'mlp_fit': (50, 'def fit_mlp_with_validation('),
}


def notebook_cell_source(cell_name):
    """Resolve a named cell and fail clearly if the notebook layout has changed."""
    index, expected_fragment = NOTEBOOK_CELLS[cell_name]
    cells = NOTEBOOK_DOCUMENT['cells']
    message = (
        f"Notebook structure changed: {cell_name!r} expects a code cell at "
        f"zero-based index {index} containing {expected_fragment!r}. "
        "Review the notebook and update NOTEBOOK_CELLS in test_data_leakage.py."
    )
    if index >= len(cells) or cells[index]['cell_type'] != 'code':
        raise AssertionError(message)
    code = ''.join(cells[index]['source'])
    if expected_fragment not in code:
        raise AssertionError(message)
    return code


def execute_notebook_cell(cell_name, scope, skip_imports=()):
    """Run production cell logic with caller-supplied offline substitutes."""
    # Skip Colab installation commands and selected heavy imports so they cannot
    # replace the substitutes supplied in scope. Keep the pipeline logic intact.
    code = '\n'.join(line for line in notebook_cell_source(cell_name).splitlines()
                     if not line.lstrip().startswith(('!', '%')))
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom)
        and any((node.module or '').startswith(prefix) for prefix in skip_imports)
    )]
    with redirect_stdout(io.StringIO()):
        exec(compile(tree, str(NOTEBOOK_PATH), 'exec'), scope)


def offline_pad_sequences(rows, maxlen, padding, truncating):
    """Implement only the notebook's post-padding mode without TensorFlow."""
    assert padding == truncating == 'post'
    result = np.zeros((len(rows), maxlen), dtype=np.int32)
    for i, row in enumerate(rows):
        result[i, :min(len(row), maxlen)] = row[:maxlen]
    return result


class TestDataLeakageRegression(unittest.TestCase):
    def build_pipeline_fixture(self, size=100, unchanged=False):
        """Exercise the real split and augmentation cells on synthetic examples."""
        frame = pd.DataFrame({
            'id': np.arange(size), 'product_uid': np.arange(size),
            'search_term': ['hammer'] * size,
            'product_title': ['steel tool'] * size,
            'relevance': 1 + np.arange(size) % 3,
        })
        _, val_indices = train_test_split(np.arange(size), test_size=0.2, random_state=42)
        frame.loc[val_indices, 'search_term'] = 'validationexclusive ☃'
        descriptions = pd.DataFrame({
            'product_uid': np.arange(size), 'product_description': ['durable'] * size
        })

        # Exercise both successful translation and the original-text fallback
        # without sending any text to a translation service.
        class Translator:
            def translate(self, text, dest):
                if unchanged:
                    raise RuntimeError('Simulated translation failure')
                return type('Translation', (), {'text': dest + ' ' + text})()

        scope = {'pd': pd, 'np': np, 'train': frame,
                 'product_descriptions': descriptions,
                 'LANGUAGES': dict.fromkeys(['en', 'es', 'fr', 'de']),
                 'translator': Translator(), 'tqdm': lambda rows, **kwargs: rows,
                 'pad_sequences': offline_pad_sequences}
        for cell_name in ['safe_translate', 'augment_row', 'split_originals']:
            execute_notebook_cell(cell_name, scope)
        validation_before = scope['validation_original'].copy(deep=True)
        for cell_name in ['generate_augmentations', 'augmentation_frame',
                          'combine_training', 'partition_counts', 'character_features']:
            execute_notebook_cell(cell_name, scope, ('tensorflow',))
        pd.testing.assert_frame_equal(scope['validation_original'], validation_before)
        return scope

    def test_augmentation_isolation_and_empty_sample(self):
        for size in [10, 100]:
            for unchanged in [False, True]:
                with self.subTest(size=size, unchanged=unchanged):
                    scope = self.build_pipeline_fixture(size, unchanged)
                    augmented = scope['train_augmented']
                    original = scope['train_original']
                    validation = scope['validation_original']
                    self.assertTrue(set(augmented.id).isdisjoint(validation.id))
                    self.assertEqual(set(augmented.id), set(original.id))
                    self.assertEqual(len(augmented), len(original) + 4 * len(scope['train_sample']))
                    pd.testing.assert_frame_equal(
                        augmented.iloc[:len(original)].reset_index(drop=True),
                        original.reset_index(drop=True))
                    for row_id in scope['train_sample'].id:
                        self.assertEqual((augmented.id == row_id).sum(), 5)
                    np.testing.assert_array_equal(scope['y_val'], validation.relevance)
                    np.testing.assert_array_equal(scope['y_train'], augmented.relevance)

    def test_training_only_vocabulary_and_feature_alignment(self):
        scope = self.build_pipeline_fixture()
        self.assertNotIn('☃', scope['char2idx'])
        self.assertEqual(scope['encode_text']('☃', scope['char2idx']), [1])
        self.assertTrue((scope['X_query_val'] == 1).any())
        for cell_name in ['tfidf_features', 'tfidf_partitions', 'ridge_fit', 'ridge_metrics']:
            execute_notebook_cell(cell_name, scope)
        self.assertNotIn('validationexclusive', scope['tfidf_vectorizer'].vocabulary_)
        self.assertEqual(scope['X_train'].shape[0], len(scope['y_train']))
        self.assertEqual(scope['X_val'].shape[0], len(scope['y_val']))

        # Execute Word2Vec's actual corpus construction with a recording model.
        for frame in [scope['train_prepared'], scope['validation_prepared']]:
            frame['search_term_tokens'] = frame.search_term.str.split()
            frame['product_text_tokens'] = frame.product_text.str.split()

        class Word2Vec:
            def __init__(self, sentences, **kwargs):
                scope['captured_corpus'] = sentences
                self.wv = {}

        scope['Word2Vec'] = Word2Vec
        execute_notebook_cell('word2vec_fit', scope, ('gensim',))
        words = {word for row in scope['captured_corpus'] for word in row}
        self.assertNotIn('validationexclusive', words)
        self.assertNotIn('☃', words)

        # Deterministic text features test alignment and scaling, not embedding
        # quality, and avoid loading a pretrained Sentence-BERT model.
        class Encoder:
            def __init__(self, name):
                pass

            def encode(self, texts, **kwargs):
                return np.array([[len(text), text.count(' ')] for text in texts], dtype=float)

        scope['SentenceTransformer'] = Encoder
        for cell_name in ['sentence_features', 'sentence_partitions', 'scale_features']:
            execute_notebook_cell(cell_name, scope, ('sentence_transformers',))
        np.testing.assert_allclose(scope['scaler'].mean_, scope['sbert_train'].mean(axis=0))
        np.testing.assert_allclose(scope['X_val'], scope['scaler'].transform(scope['sbert_val']))
        self.assertEqual(scope['X_train'].shape, (len(scope['y_train']), 8))
        self.assertEqual(scope['X_val'].shape, (len(scope['y_val']), 8))

    def test_lstm_explicit_validation(self):
        scope = self.build_pipeline_fixture()
        self.prepare_word_features(scope)

        # Record the real fit call's inputs without constructing or training an LSTM.
        class Model:
            def fit(self, inputs, labels, **kwargs):
                self.inputs, self.labels, self.kwargs = inputs, labels, kwargs

        scope['EarlyStopping'] = lambda **kwargs: kwargs
        for cell_name, name in [('first_lstm_fit', 'model1'), ('second_lstm_fit', 'model2')]:
            model = Model()
            scope[name] = model
            execute_notebook_cell(cell_name, scope, ('tensorflow',))
            prefix = 'X_word_' if name == 'model2' else 'X_'
            self.assertIs(model.labels, scope['y_train'])
            self.assertIs(model.inputs[0], scope[prefix + 'query_train'])
            self.assertIs(model.inputs[1], scope[prefix + 'product_train'])
            validation_inputs, labels = model.kwargs['validation_data']
            self.assertIs(validation_inputs[0], scope[prefix + 'query_val'])
            self.assertIs(validation_inputs[1], scope[prefix + 'product_val'])
            self.assertIs(labels, scope['y_val'])
            self.assertNotIn('validation_split', model.kwargs)
            self.assertEqual(model.kwargs['epochs'], 10)
            self.assertEqual(model.kwargs['callbacks'], [dict(
                monitor='val_loss', patience=3, restore_best_weights=True)])

    def prepare_word_features(self, scope):
        # Supply known vectors to verify ID-to-vector alignment without gensim.
        class WordVectors:
            index_to_key = ['hammer', 'steel', 'tool', 'durable']
            vector_size = 100

            def __getitem__(self, word):
                return np.full(self.vector_size, self.index_to_key.index(word) + 1,
                               dtype=np.float32)

        scope['w2v_model'] = type('Word2VecStub', (), {'wv': WordVectors()})()
        for frame in [scope['train_prepared'], scope['validation_prepared']]:
            frame['search_term_tokens'] = frame.search_term.str.split()
            frame['product_text_tokens'] = frame.product_text.str.split()
        execute_notebook_cell('word_features', scope, ('tensorflow',))

    def test_word2vec_embedding_and_word_inputs(self):
        scope = self.build_pipeline_fixture()
        self.prepare_word_features(scope)
        matrix = scope['word_embedding_matrix']
        np.testing.assert_array_equal(matrix[0], np.zeros(100))
        self.assertNotIn('validationexclusive', scope['word2idx'])
        self.assertEqual(scope['X_word_query_val'][0, 0], scope['WORD_UNK_ID'])
        for word, index in scope['word2idx'].items():
            np.testing.assert_array_equal(matrix[index], scope['w2v_model'].wv[word])
        for partition, labels in [('train', 'y_train'), ('val', 'y_val')]:
            self.assertEqual(scope['X_word_query_' + partition].shape, (len(scope[labels]), 40))
            self.assertEqual(scope['X_word_product_' + partition].shape, (len(scope[labels]), 400))
        samples = pd.DataFrame({
            'search_term_tokens': [[], ['hammer'] * 45],
            'product_text_tokens': [[], ['steel'] * 405],
        })
        queries, products = scope['encode_word_partition'](samples)
        self.assertFalse(queries[0].any())
        self.assertFalse(products[0].any())
        self.assertTrue((queries[1] == scope['word2idx']['hammer']).all())
        self.assertTrue((products[1] == scope['word2idx']['steel']).all())

        # Record the actual model-building calls: both branches must invoke the
        # same word encoder, initialized with the Word2Vec matrix.
        embedding_calls, encoder_inputs = [], []

        def embedding(**kwargs):
            embedding_calls.append(kwargs)
            return lambda inputs: object()

        class SharedEncoder:
            def __call__(self, inputs):
                encoder_inputs.append(inputs)
                return object()

        scope.update({
            'Input': lambda **kwargs: kwargs,
            'Embedding': embedding,
            'Dropout': lambda *args: lambda inputs: object(),
            'LSTM': lambda *args: lambda inputs: object(),
            'Model': lambda *args, **kwargs: SharedEncoder(),
            'Dense': lambda *args, **kwargs: lambda inputs: object(),
            'Lambda': lambda *args: lambda inputs: object(),
            'Concatenate': lambda: lambda inputs: object(),
        })
        execute_notebook_cell('word_model', scope, ('tensorflow',))
        self.assertEqual(len(embedding_calls), 1)
        embedding_config = embedding_calls[0]
        self.assertIs(embedding_config['weights'][0], matrix)
        self.assertEqual(embedding_config['output_dim'], 100)
        self.assertTrue(embedding_config['mask_zero'])
        self.assertTrue(embedding_config['trainable'])
        self.assertEqual(encoder_inputs, [scope['query_input'], scope['product_input']])
        self.assertEqual([item['shape'] for item in encoder_inputs], [(40,), (400,)])

    def test_mlp_stopping_and_checkpoint(self):
        # Load the actual early-stopping helper without starting model training.
        # Scripted scores make patience and checkpoint behavior deterministic.
        tree = ast.parse(notebook_cell_source('mlp_fit'))
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        scope = {'np': np}
        exec(compile(tree, str(NOTEBOOK_PATH), 'exec'), scope)
        fit = scope['fit_mlp_with_validation']

        class Regressor:
            early_stopping = False

            def __init__(self, scores):
                self.scores, self.epoch = scores, 0

            def partial_fit(self, X, y):
                self.epoch += 1

            def predict(self, X):
                return self.scores[min(self.epoch - 1, len(self.scores) - 1)]

        scope['r2_score'] = lambda y, prediction: prediction
        model = Regressor([0.5, 0.8, 0.7])
        best, scores = fit(model, None, None, None, None)
        self.assertEqual(len(scores), 12)
        self.assertEqual(best.epoch, 2)
        self.assertIsNot(best, model)
        best, scores = fit(Regressor([0.5, 0.50001]), None, None, None, None, patience=2)
        self.assertEqual(len(scores), 3)
        self.assertEqual(best.epoch, 2)
        _, scores = fit(Regressor([0.5]), None, None, None, None, max_epochs=3)
        self.assertEqual(len(scores), 3)
        with self.assertRaises(ValueError):
            fit(Regressor([np.nan]), None, None, None, None)

        # Smoke-test the complete notebook MLP cell using real scikit-learn.
        rng = np.random.default_rng(42)
        scope = {'np': np, 'X_train': rng.normal(size=(80, 8)),
                 'X_val': rng.normal(size=(20, 8))}
        scope['y_train'] = scope['X_train'][:, 0] + 2
        scope['y_val'] = scope['X_val'][:, 0] + 2
        execute_notebook_cell('mlp_fit', scope)
        from sklearn.metrics import r2_score
        self.assertAlmostEqual(
            r2_score(scope['y_val'], scope['regressor'].predict(scope['X_val'])),
            max(scope['validation_scores']))
        self.assertLessEqual(len(scope['validation_scores']), 200)

    def test_notebook_structure_and_syntax(self):
        nbformat.validate(nbformat.read(NOTEBOOK_PATH, as_version=4))
        for cell_name in NOTEBOOK_CELLS:
            notebook_cell_source(cell_name)
        all_code = ''
        for cell in NOTEBOOK_DOCUMENT['cells']:
            if cell['cell_type'] != 'code':
                continue
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
            code = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                             if not line.lstrip().startswith(('!', '%')) and line.strip() != 'ls')
            ast.parse(code)
            all_code += code + '\n'
        self.assertEqual(all_code.count('= train_test_split('), 1)
        self.assertNotIn('validation_split=', all_code)
        self.assertNotIn('validation_fraction=', all_code)


if __name__ == '__main__':
    unittest.main()

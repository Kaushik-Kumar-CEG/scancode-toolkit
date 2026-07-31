# tests for train_model.py and export_onnx.py
# these cover the pure logic, the heavy model parts are exercised on a gpu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_model import align_labels
from train_model import decode_row
from train_model import extract_spans
from train_model import first_subword_positions
from train_model import IGNORE_INDEX
from train_model import LABEL2ID
from train_model import LABELS


class FakeEncoding(dict):
    """Mimics a tokenizer encoding with a fixed word_ids mapping"""

    def __init__(self, word_ids):
        super().__init__()
        self._word_ids = word_ids
        self['input_ids'] = [0] * len(word_ids)
        self['attention_mask'] = [1] * len(word_ids)

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    def __init__(self, word_ids):
        self._word_ids = word_ids

    def __call__(self, tokens, **kwargs):
        return FakeEncoding(self._word_ids)


class TestExtractSpans:

    def test_single_phrase(self):
        assert extract_spans(['O', 'B-REQ', 'I-REQ', 'E-REQ', 'O']) == {(1, 3)}

    def test_single_token_phrase(self):
        assert extract_spans(['O', 'S-REQ', 'O']) == {(1, 1)}

    def test_two_phrases(self):
        assert extract_spans(['S-REQ', 'O', 'B-REQ', 'E-REQ']) == {(0, 0), (2, 3)}

    def test_no_phrase(self):
        assert extract_spans(['O', 'O', 'O']) == set()

    def test_handles_inside_without_begin(self):
        # raw softmax output can be malformed, an I with no B still yields a span
        assert extract_spans(['O', 'I-REQ', 'E-REQ']) == {(1, 2)}

    def test_handles_unterminated_begin(self):
        assert extract_spans(['O', 'B-REQ', 'I-REQ']) == {(1, 2)}


class TestDecodeRow:

    def test_drops_ignored_positions(self):
        preds = [1, 0, 2, 0]
        labels = [1, IGNORE_INDEX, 2, IGNORE_INDEX]
        pred_tags, true_tags = decode_row(preds, labels)
        assert true_tags == ['B-REQ', 'I-REQ']
        assert pred_tags == ['B-REQ', 'I-REQ']

    def test_unknown_prediction_falls_back_to_o(self):
        pred_tags, true_tags = decode_row([99], [0])
        assert true_tags == ['O']
        assert pred_tags == ['O']


class TestAlignLabels:

    def test_first_subword_keeps_label(self):
        # CLS Apache Lic ##ense SEP, Apache is one subword, License is two
        tokenizer = FakeTokenizer([None, 0, 1, 1, None])
        encoding = align_labels(['Apache', 'License'], ['B-REQ', 'E-REQ'], tokenizer, 512)
        assert encoding['labels'] == [
            IGNORE_INDEX, LABEL2ID['B-REQ'], LABEL2ID['E-REQ'], IGNORE_INDEX, IGNORE_INDEX,
        ]

    def test_all_outside(self):
        tokenizer = FakeTokenizer([None, 0, 1, None])
        encoding = align_labels(['the', 'license'], ['O', 'O'], tokenizer, 512)
        assert encoding['labels'] == [IGNORE_INDEX, 0, 0, IGNORE_INDEX]


class TestFirstSubwordPositions:

    def test_skips_specials_and_continuations(self):
        # CLS w0 w1a w1b w2 SEP
        assert first_subword_positions([None, 0, 1, 1, 2, None]) == [1, 2, 4]

    def test_without_special_tokens(self):
        # depending on its files a tokenizer may add no CLS or SEP at all
        assert first_subword_positions([0, 1, 2]) == [0, 1, 2]

    def test_long_word(self):
        assert first_subword_positions([None, 0, 0, 0, 1, None]) == [1, 4]

    def test_nothing_to_tag(self):
        assert first_subword_positions([None, None]) == []
        assert first_subword_positions([]) == []


def make_crf_tagger():
    """A PhraseTagger with only the CRF built, the backbone needs a download"""
    import torch
    from torchcrf import CRF
    from phrase_model import PhraseTagger

    tagger = PhraseTagger.__new__(PhraseTagger)
    torch.nn.Module.__init__(tagger)
    tagger.use_crf = True
    tagger.num_labels = len(LABELS)
    tagger.crf = CRF(len(LABELS), batch_first=True)
    # zero transitions make the decode a plain argmax, so the expected path is obvious
    with torch.no_grad():
        for param in tagger.crf.parameters():
            param.zero_()
    return tagger


def test_predict_words_uses_the_first_subword():
    import torch

    tagger = make_crf_tagger()
    # CLS w0 w1a w1b SEP, the score on the continuation subword must be ignored
    emissions = torch.zeros((1, 5, len(LABELS)))
    emissions[0, 1, LABEL2ID['B-REQ']] = 9.0
    emissions[0, 2, LABEL2ID['E-REQ']] = 9.0
    emissions[0, 3, LABEL2ID['S-REQ']] = 9.0
    tagger.emissions = lambda input_ids, attention_mask: emissions

    ids = torch.zeros((1, 5), dtype=torch.long)
    tags = tagger.predict_words(ids, ids, [None, 0, 1, 1, None])
    assert tags == [LABEL2ID['B-REQ'], LABEL2ID['E-REQ']]


def test_predict_words_with_no_words():
    import torch

    tagger = make_crf_tagger()
    tagger.emissions = lambda input_ids, attention_mask: torch.zeros((1, 2, len(LABELS)))
    ids = torch.zeros((1, 2), dtype=torch.long)
    assert tagger.predict_words(ids, ids, [None, None]) == []


def test_viterbi_with_zero_transitions_is_argmax():
    import numpy as np
    from export_onnx import viterbi_decode

    emissions = np.array([[0.1, 0.9, 0.0, 0.0, 0.0],
                          [0.7, 0.2, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.9]])
    zeros_t = np.zeros((5, 5))
    zeros_v = np.zeros(5)
    path = viterbi_decode(emissions, zeros_v, zeros_t, zeros_v)
    assert path == [1, 0, 4]


def test_viterbi_avoids_forbidden_transition():
    import numpy as np
    from export_onnx import viterbi_decode

    # emissions want tag 1 then tag 0, but 1 -> 0 is heavily penalised
    emissions = np.array([[0.0, 1.0], [1.0, 0.0]])
    transitions = np.array([[0.0, 0.0], [-100.0, 0.0]])
    zeros_v = np.zeros(2)
    path = viterbi_decode(emissions, zeros_v, transitions, zeros_v)
    assert path[0] == path[1]


def test_compute_metrics_perfect_and_exact():
    from train_model import compute_metrics

    # two rules, word level predictions and labels line up perfectly
    predictions = [[1, 3, 0], [4, IGNORE_INDEX, IGNORE_INDEX]]
    labels = [[1, 3, 0], [4, IGNORE_INDEX, IGNORE_INDEX]]
    scores = compute_metrics((predictions, labels))
    assert scores['f1'] == 1.0
    assert scores['exact_match'] == 1.0


def test_compute_metrics_counts_a_miss():
    from train_model import compute_metrics

    # predict O everywhere, the gold has one phrase, so f1 is 0
    predictions = [[0, 0, 0]]
    labels = [[1, 3, 0]]
    scores = compute_metrics((predictions, labels))
    assert scores['f1'] == 0.0
    assert scores['exact_match'] == 0.0

import pytest
from src.business_entity_resolution.evaluation.f05 import calculate_macro_f05

def test_exact_match():
    y_true = {'s1': {'s2', 's3'}}
    y_pred = {'s1': {'s2', 's3'}}
    assert calculate_macro_f05(y_true, y_pred) == 1.0

def test_missing_match():
    y_true = {'s1': {'s2', 's3'}}
    y_pred = {'s1': {'s2'}}
    # precision = 1.0, recall = 0.5
    # f05 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333...
    assert abs(calculate_macro_f05(y_true, y_pred) - 0.8333333333) < 1e-6

def test_extra_false_positive():
    y_true = {'s1': {'s2'}}
    y_pred = {'s1': {'s2', 's3'}}
    # precision = 0.5, recall = 1.0
    # f05 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
    assert abs(calculate_macro_f05(y_true, y_pred) - 0.5555555555) < 1e-6

def test_singleton_correct():
    y_true = {'s1': set()}
    y_pred = {'s1': set()}
    assert calculate_macro_f05(y_true, y_pred) == 1.0

def test_singleton_incorrect():
    y_true = {'s1': set()}
    y_pred = {'s1': {'s2'}}
    assert calculate_macro_f05(y_true, y_pred) == 0.0

def test_empty_prediction_against_non_empty_truth():
    y_true = {'s1': {'s2'}}
    y_pred = {'s1': set()}
    assert calculate_macro_f05(y_true, y_pred) == 0.0

def test_macro_average():
    y_true = {
        's1_1': {'s2_1'},       # Exact match
        's1_2': set(),          # Singleton correct
        's1_3': {'s2_3'},       # Missing prediction
        's1_4': set()           # Singleton incorrect
    }
    y_pred = {
        's1_1': {'s2_1'},       # score = 1.0
        's1_2': set(),          # score = 1.0
        's1_3': set(),          # score = 0.0
        's1_4': {'s2_4'}        # score = 0.0
    }
    # average of 1.0, 1.0, 0.0, 0.0 is 0.5
    assert calculate_macro_f05(y_true, y_pred) == 0.5

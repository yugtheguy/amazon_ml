import pytest
import yaml

def test_e002_gpu_config():
    with open("configs/e002_word_gpu.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    assert "retrieval" in config
    assert config["retrieval"]["method"] == "gpu_word_tfidf"
    assert config["retrieval"]["analyzer"] == "word"
    assert config["retrieval"]["ngram_min"] == 1
    assert config["retrieval"]["ngram_max"] == 2

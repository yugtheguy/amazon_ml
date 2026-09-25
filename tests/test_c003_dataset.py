import pytest
import pandas as pd
from src.business_entity_resolution.features.validation import validate_shard_s1_isolation

def test_validate_shard_s1_isolation():
    global_seen = {"A", "B", "C"}
    
    # Valid - entirely disjoint S1 entities
    validate_shard_s1_isolation({"D", "E"}, global_seen, "test_shard")
    assert global_seen == {"A", "B", "C", "D", "E"}
    
    # Invalid - overlap
    with pytest.raises(ValueError, match="S1 Isolation violation"):
        validate_shard_s1_isolation({"E", "F"}, global_seen, "test_shard_2")

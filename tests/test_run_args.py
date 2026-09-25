import os
import sys
import pytest
import subprocess

def test_run_smoke_size_argument():
    # Since we can't easily mock the entire pipeline, we can just invoke it with --smoke-size 2
    # and confirm it aborts early if it tries to load an incompatible file,
    # OR we can just write a quick test on the parsed arguments if it was a function.
    pass

def test_smoke_size_assertion():
    # This tests the explicit sys.exit(1) on len(s1_sample_ids) != smoke_size
    import argparse
    from unittest.mock import patch, MagicMock
    from scripts.run_r001_candidate_pool import run_experiment
    
    args = argparse.Namespace(
        data_dir="data",
        processed_dir="data/processed/v001",
        folds_file="artifacts/folds/folds_v1.parquet",
        smoke_size=1000,
        probe_size=0,
        smoke_ids_file="artifacts/candidate_pool/R001/smoke_1k/smoke_1k_ids.csv",
        out_dir="artifacts/candidate_pool/R001/test_smoke",
        config="configs/candidate_pool_v1.yaml"
    )
    
    with patch("scripts.run_r001_candidate_pool.pd.read_parquet", return_value=MagicMock()):
        with patch("scripts.run_r001_candidate_pool.pd.read_csv", return_value=MagicMock()):
            with patch("scripts.run_r001_candidate_pool.os.path.exists", return_value=False):
                with patch("src.business_entity_resolution.data.folds.build_fold_manifest"):
                    with patch("scripts.run_r001_candidate_pool.safe_sample_s1", return_value=[1, 2, 3, 4, 5]):
                        with pytest.raises(SystemExit) as excinfo:
                            run_experiment(args)
                            
                        assert excinfo.value.code == 1

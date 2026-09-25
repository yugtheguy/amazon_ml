import pandas as pd
import json

def update_log():
    log_path = "experiments/experiment_log.csv"
    log = pd.read_csv(log_path)
    
    with open("artifacts/retrieval/E000/decision_metrics_D0.json") as f:
        d0_f05 = json.load(f)["macro_f05"]
        
    with open("artifacts/retrieval/E000/decision_metrics_D1.json") as f:
        d1_f05 = json.load(f)["macro_f05"]
        
    best_policy = "D0" if d0_f05 >= d1_f05 else "D1"
    best_f05 = max(d0_f05, d1_f05)
    
    new_entry = pd.DataFrame([{
        "experiment_id": "E000",
        "description": f"Exact deterministic retrieval + {best_policy} conservative decision",
        "architecture_stage": "retrieval_and_decision",
        "val_macro_f05": best_f05,
        "test_macro_f05": None,
        "notes": "First baseline, conservative rule gating, exact deterministic channels"
    }])
    
    log = pd.concat([log, new_entry], ignore_index=True)
    log.to_csv(log_path, index=False)
    print(f"Updated {log_path} with E000.")

if __name__ == "__main__":
    update_log()

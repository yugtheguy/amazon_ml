def calculate_macro_f05(y_true: dict, y_pred: dict) -> float:
    """
    Calculates the macro F0.5 score at the entity level.
    
    y_true: dict mapping source1_entity_id -> set of matched_entity_ids (empty set for singletons)
    y_pred: dict mapping source1_entity_id -> set of predicted matched_entity_ids
    
    Returns the macro average F0.5 across all entities in y_true.
    """
    if not y_true:
        return 0.0
        
    f05_scores = []
    
    for s1_id, true_matches in y_true.items():
        pred_matches = y_pred.get(s1_id, set())
        
        if len(true_matches) == 0:
            # Singleton handling
            if len(pred_matches) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
        else:
            # Non-singleton handling
            if len(pred_matches) == 0:
                f05_scores.append(0.0)
            else:
                tp = len(true_matches.intersection(pred_matches))
                fp = len(pred_matches - true_matches)
                fn = len(true_matches - pred_matches)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if precision == 0.0 and recall == 0.0:
                    f05_scores.append(0.0)
                else:
                    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                    f05_scores.append(f05)
                    
    return sum(f05_scores) / len(f05_scores)

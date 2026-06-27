import numpy as np
import pandas as pd

def resolve_knockout_draw(probs):
    """Normalizes Win/Loss probabilities to break a knockout draw (Option A)."""
    total_win_p = probs[0] + probs[2]
    
    # Safety net: If the model is somehow 100% sure it's a draw, or values are NaN/0
    if total_win_p <= 0 or np.isnan(total_win_p):
        return np.random.choice([0, 2], p=[0.5, 0.5])
        
    # Cast directly to float64 and normalize the array in-place.
    # This completely bypasses the 32-bit/64-bit XGBoost precision crash.
    p_array = np.array([probs[0], probs[2]], dtype=np.float64)
    p_array /= p_array.sum()
    
    return np.random.choice([0, 2], p=p_array)


def simulate_single_match(model, home, away, match_weight, is_neutral, teams_state, h2h_lookup):
    # Construct features dynamically and updates team form
    home_rank = teams_state[home]['rank']
    away_rank = teams_state[away]['rank']
    rank_diff = home_rank - away_rank
    
    home_form = np.mean(teams_state[home]['form_history'])
    away_form = np.mean(teams_state[away]['form_history'])
    
    # Static H2H lookup map
    h2h_diff = h2h_lookup.get((home, away), 0.0)
    
    # Assemble feature row for XGBoost
    match_features = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'rank_diff': rank_diff,
        'point_diff': 0.0,
        'home_fifa_rank': home_rank,
        'home_recent_form': home_form,
        'away_recent_form': away_form,
        'h2h_goal_diff': h2h_diff,
        'match_weight': match_weight,
        'is_neutral': is_neutral
    }])
    
    match_features['home_team'] = match_features['home_team'].astype('category')
    match_features['away_team'] = match_features['away_team'].astype('category')
    
    probs = model.predict_proba(match_features)[0]
    outcome = np.random.choice([0, 1, 2], p=probs)
    
    # Dynamic form updates (sliding window of 5 matches)
    if outcome == 0:
        teams_state[home]['form_history'].append(3)
        teams_state[away]['form_history'].append(0)
    elif outcome == 1:
        teams_state[home]['form_history'].append(1)
        teams_state[away]['form_history'].append(1)
    else:
        teams_state[home]['form_history'].append(0)
        teams_state[away]['form_history'].append(3)
        
    teams_state[home]['form_history'].pop(0)
    teams_state[away]['form_history'].pop(0)
    
    return outcome, probs
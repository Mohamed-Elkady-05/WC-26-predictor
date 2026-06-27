import copy
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.sims.bracket import resolve_knockout_draw, simulate_single_match


def run_tournament_loop(model, future_fixtures, base_df, h2h_lookup, initial_teams_state):
    teams_state = copy.deepcopy(initial_teams_state)
    # --- 1. GROUP STAGE ---
    group_matches = future_fixtures[future_fixtures['stage'] == 'Group Stage']
    groups = group_matches['group'].unique()
    
    standings = {}
    for group in groups:
        standings[group] = {}
        teams_in_group = group_matches[group_matches['group'] == group]['home_team'].unique()
        for team in teams_in_group:
            # Add probabilistic tie-breaker score (inverse rank + normal noise)
            tie_breaker = (1.0 / teams_state[team]['rank']) + np.random.normal(0, 0.01)
            standings[group][team] = {'points': 0, 'tie_breaker': tie_breaker}
            
    for _, match in group_matches.iterrows():
        g = match['group']
        h, a = match['home_team'], match['away_team']
        res, _ = simulate_single_match(model, h, a, match['match_weight'], match['is_neutral'], teams_state, h2h_lookup)
        
        if res == 0:
            standings[g][h]['points'] += 3
        elif res == 1:
            standings[g][h]['points'] += 1
            standings[g][a]['points'] += 1
        else:
            standings[g][a]['points'] += 3
            
    advancing_teams = []
    third_place_pool = []
    
    for g in groups:
        sorted_teams = sorted(
            standings[g].items(), 
            key=lambda item: (item[1]['points'], item[1]['tie_breaker']), 
            reverse=True
        )
        advancing_teams.append(sorted_teams[0][0])
        advancing_teams.append(sorted_teams[1][0])
        third_place_pool.append((sorted_teams[2][0], sorted_teams[2][1]))
        
    sorted_third_places = sorted(third_place_pool, key=lambda item: (item[1]['points'], item[1]['tie_breaker']), reverse=True)
    for i in range(8):
        advancing_teams.append(sorted_third_places[i][0])
        
    # --- 2. KNOCKOUT STAGE ---
    current_knockout_teams = advancing_teams.copy()
    round_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    
    for round_name in round_names:
        next_round_winners = []
        for i in range(0, len(current_knockout_teams), 2):
            h = current_knockout_teams[i]
            a = current_knockout_teams[i+1]
            
            res, probs = simulate_single_match(model, h, a, 60, True, teams_state, h2h_lookup)
            
            if res == 1:
                res = resolve_knockout_draw(probs)
                
            winner = h if res == 0 else a
            next_round_winners.append(winner)
            
        current_knockout_teams = next_round_winners
        
    return current_knockout_teams[0]

def execute_simulation(num_simulations=10000):
    print("Initializing components for Monte Carlo Simulation...")
    model = joblib.load('outputs/models/xgboost_model.pkl')
    future_fixtures = pd.read_csv('data/processed/future_fixtures.csv')
    base_df = pd.read_csv('data/processed/training_features.csv')
    
    h2h_lookup = base_df.groupby(['home_team', 'away_team'])['h2h_goal_diff'].last().to_dict()
    latest_team_data = base_df.sort_values('date').groupby('home_team').last()
    
    initial_teams_state = {}
    for team in base_df['home_team'].unique():
        if team in latest_team_data.index:
            form_val = latest_team_data.loc[team, 'home_recent_form']
            if pd.isna(form_val): form_val = 1.0
            initial_teams_state[team] = {
                'rank': int(latest_team_data.loc[team, 'home_fifa_rank']),
                'form_history': [form_val] * 5
            }
            
    trophy_tally = {}
    print(f"Running {num_simulations} iterations...")
    for sim in tqdm(range(num_simulations), desc="Simulating Tournaments"):
        champion = run_tournament_loop(model, future_fixtures, base_df, h2h_lookup, initial_teams_state)
        trophy_tally[champion] = trophy_tally.get(champion, 0) + 1
        
    leaderboard = pd.DataFrame(list(trophy_tally.items()), columns=['Country', 'Titles'])
    leaderboard['Win Probability (%)'] = (leaderboard['Titles'] / num_simulations) * 100
    leaderboard = leaderboard.sort_values(by='Titles', ascending=False).reset_index(drop=True)
    
    leaderboard.index = leaderboard.index + 1
    leaderboard.to_csv('outputs/results/simulation_leaderboard.csv', index=False)
    print("\nSimulation complete. Leaderboard saved to outputs/results/simulation_leaderboard.csv")
    print(leaderboard.head(10))
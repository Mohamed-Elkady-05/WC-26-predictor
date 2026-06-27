import pandas as pd
from collections import defaultdict

print("Loading data...")
fixtures = pd.read_csv('data/processed/future_fixtures.csv')

# 1. Add the missing 'stage' column
fixtures['stage'] = 'Group Stage'

# 2. Build a web of matchups to see who plays who
matchups = defaultdict(set)
for _, row in fixtures.iterrows():
    matchups[row['home_team']].add(row['away_team'])
    matchups[row['away_team']].add(row['home_team'])

# 3. Find connected groups using Breadth-First Search (BFS)
visited = set()
discovered_groups = []

for team in matchups.keys():
    if team not in visited:
        current_group = []
        queue = [team]
        
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                current_group.append(current)
                # Add all opponents to the queue
                for opponent in matchups[current]:
                    if opponent not in visited:
                        queue.append(opponent)
                        
        discovered_groups.append(current_group)

# 4. Name them "Group A", "Group B", etc., and map them to the teams
team_to_group = {}
for i, group_teams in enumerate(discovered_groups):
    group_name = f"Group {chr(65 + i)}"  # chr(65) is 'A', chr(66) is 'B', etc.
    for team in group_teams:
        team_to_group[team] = group_name

# 5. Apply the mapping to your fixtures
fixtures['group'] = fixtures['home_team'].map(team_to_group)

# 6. Save the fixed file
fixtures.to_csv('data/processed/future_fixtures.csv', index=False)

print(f"Fix complete! Successfully discovered {len(discovered_groups)} groups automatically.")
print("'stage' and 'group' columns have been added.")
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.elo import EloRating

def replay_history(input_path="data/processed/multi_league_master.csv", 
                   output_path="data/processed/multi_league_master_peopled.csv"):
    """
    Replays history chronologically to calculate dynamic ELO and Form features.
    """
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure chronological order
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize ELO and Form
    elo_tracker = EloRating(k_factor=20, home_advantage=100)
    # Form state: Dict[team] = List of last 5 outcomes {points, goals_for, goals_against}
    forms = {} 

    def get_form_avg(team):
        if team not in forms or not forms[team]:
            return 1.5, 1.5, 1.5 # Defaults
        recent = forms[team][-5:]
        n = len(recent)
        pts = sum(m['pts'] for m in recent) / n
        gf = sum(m['gf'] for m in recent) / n
        ga = sum(m['ga'] for m in recent) / n
        return pts, gf, ga

    def update_form(team, pts, gf, ga):
        if team not in forms: forms[team] = []
        forms[team].append({'pts': pts, 'gf': gf, 'ga': ga})

    # Prepare historical columns
    new_data = []

    print("Replaying 7000 matches...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ht = row['HomeTeam']
        at = row['AwayTeam']
        res = row['FTR']
        hg = row['FTHG']
        ag = row['FTAG']
        
        # 1. GET STATE BEFORE MATCH
        h_elo = elo_tracker.get_rating(ht)
        a_elo = elo_tracker.get_rating(at)
        
        h_pts_avg, h_gf_avg, h_ga_avg = get_form_avg(ht)
        a_pts_avg, a_gf_avg, a_ga_avg = get_form_avg(at)
        
        # Update row with dynamic features
        row_dict = row.to_dict()
        row_dict['HomeEloBefore'] = h_elo
        row_dict['AwayEloBefore'] = a_elo
        row_dict['EloDiff'] = h_elo - a_elo
        
        row_dict['Home_Pts_L5'] = h_pts_avg
        row_dict['Home_GF_L5'] = h_gf_avg
        row_dict['Home_GA_L5'] = h_ga_avg
        
        row_dict['Away_Pts_L5'] = a_pts_avg
        row_dict['Away_GF_L5'] = a_gf_avg
        row_dict['Away_GA_L5'] = a_ga_avg
        
        # We assume overall form = specific form for this simple replay
        row_dict['Home_Overall_Pts_L5'] = h_pts_avg
        row_dict['Home_Overall_GF_L5'] = h_gf_avg
        row_dict['Home_Overall_GA_L5'] = h_ga_avg
        row_dict['Away_Overall_Pts_L5'] = a_pts_avg
        row_dict['Away_Overall_GF_L5'] = a_gf_avg
        row_dict['Away_Overall_GA_L5'] = a_ga_avg
        
        # Streaks (approximated)
        # Note: True streak needs more logic, but even basic form is huge improvement over nothing
        row_dict['WinStreak_L5'] = 1.0 if (len(forms.get(ht, [])) > 0 and forms[ht][-1]['pts'] == 3) else 0.0
        row_dict['LossStreak_L5'] = 1.0 if (len(forms.get(ht, [])) > 0 and forms[ht][-1]['pts'] == 0) else 0.0
        row_dict['DaysSinceLastMatch'] = 7.0 # Default
        
        new_data.append(row_dict)
        
        # 2. UPDATE STATE AFTER MATCH
        elo_tracker.update(ht, at, res)
        
        h_pts = 3 if res == 'H' else (1 if res == 'D' else 0)
        a_pts = 3 if res == 'A' else (1 if res == 'D' else 0)
        update_form(ht, h_pts, hg, ag)
        update_form(at, a_pts, ag, hg)

    # Save
    out_df = pd.DataFrame(new_data)
    out_df.to_csv(output_path, index=False)
    print(f"Saved enriched dataset to {output_path}")

if __name__ == "__main__":
    replay_history()

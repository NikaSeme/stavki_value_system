"""
Dynamic Poisson Model (Dixon-Coles style)
Supports time-decay and incremental updates.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import poisson
import pickle
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class PoissonMatchPredictor:
    """
    Poisson-based match outcome predictor (Dixon-Coles).
    Learns team attack/defense strengths and supports dynamic updates.
    """
    
    def __init__(self, home_advantage=0.15, time_decay_rate=0.003):
        self.home_advantage = home_advantage
        self.time_decay_rate = time_decay_rate
        self.team_attack = {}  # Team -> strength (1.0 = avg)
        self.team_defense = {} # Team -> strength (1.0 = avg)
        self.league_avg_goals = 1.5
        self.league_baselines = {}
        self.team_leagues = {}
        
        # History for rebuilding/retraining if needed
        self.match_history = [] 
        
    def fit(self, df: pd.DataFrame):
        """
        Fit model on historical data.
        df must have: HomeTeam, AwayTeam, FTHG, FTAG, Date, (League)
        """
        logger.info(f"Fitting Poisson model on {len(df)} matches")
        
        # Copy to avoid side effects
        df_copy = df.copy()
        if 'Date' in df_copy.columns:
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            max_date = df_copy['Date'].max()
            df_copy['days_ago'] = (max_date - df_copy['Date']).dt.days
            df_copy['weight'] = np.exp(-self.time_decay_rate * df_copy['days_ago'])
        else:
            df_copy['weight'] = 1.0
            
        # 1. League Baselines
        total_wg = (df_copy['FTHG'] * df_copy['weight']).sum() + (df_copy['FTAG'] * df_copy['weight']).sum()
        total_w = df_copy['weight'].sum() * 2
        
        if total_w > 0:
            self.league_avg_goals = total_wg / total_w
            
        if 'League' in df_copy.columns:
            for league in df_copy['League'].unique():
                ldf = df_copy[df_copy['League'] == league]
                wg = (ldf['FTHG'] * ldf['weight']).sum() + (ldf['FTAG'] * ldf['weight']).sum()
                w = ldf['weight'].sum() * 2
                if w > 0:
                    self.league_baselines[league] = wg / w
                
                for t in set(ldf['HomeTeam']) | set(ldf['AwayTeam']):
                    self.team_leagues[t] = league
                    
        # 2. Team Strengths
        # We accumulate weighted goals for/against
        stats = defaultdict(lambda: {'gf': 0.0, 'ga': 0.0, 'w': 0.0})
        
        for _, row in df_copy.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            hg, ag = row['FTHG'], row['FTAG']
            w = row['weight']
            
            stats[h]['gf'] += hg * w
            stats[h]['ga'] += ag * w
            stats[h]['w'] += w
            
            stats[a]['gf'] += ag * w
            stats[a]['ga'] += hg * w
            stats[a]['w'] += w
            
        # Calculate strengths
        for team, s in stats.items():
            if s['w'] > 0:
                avg_gf = s['gf'] / s['w']
                avg_ga = s['ga'] / s['w']
                self.team_attack[team] = avg_gf / self.league_avg_goals
                self.team_defense[team] = avg_ga / self.league_avg_goals
            else:
                self.team_attack[team] = 1.0
                self.team_defense[team] = 1.0
                
    def update_match(self, home, away, hg, ag, date=None, league=None, weight=1.0):
        """
        Dynamically update team stats after a match.
        Simple EMA-like update or re-calculation?
        For efficiency in backtest: Simple Exponential Moving Average (EMA) update.
        
        New_Strength = Alpha * Observed + (1-Alpha) * Old_Strength
        Alpha depends on learning rate.
        
        Alternatively: Re-fit on window? Expensive.
        
        Let's use a "Partial Fit" approach. 
        We adjust the team's attack/defense slightly based on performance vs expected.
        """
        # Get current ratings
        h_att = self.team_attack.get(home, 1.0)
        h_def = self.team_defense.get(home, 1.0)
        a_att = self.team_attack.get(away, 1.0)
        a_def = self.team_defense.get(away, 1.0)
        
        baseline = self.league_avg_goals
        if league and league in self.league_baselines:
            baseline = self.league_baselines[league]
            
        # Expected goals
        # lambda_h = baseline * h_att * a_def * home_adv
        # lambda_a = baseline * a_att * h_def
        
        # We observed hg, ag.
        # If hg > lambda_h: Home Attack UP, Away Defense WORSE (Higher value)
        
        # Update rate (Learning rate)
        # Low value = stable, High value = reactive
        lr = 0.05 
        
        # Update Home Attack
        # Performance = hg / (baseline * a_def * 1.15)
        # But this is noisy.
        # Better: h_att_new = h_att + lr * (Actual - Expected) / Scaling
        
        # Simple heuristic update:
        # If scored more than expected, boost attack
        
        exp_h = baseline * h_att * a_def * (1 + self.home_advantage)
        exp_a = baseline * a_att * h_def
        
        # Error
        err_h = hg - exp_h
        err_a = ag - exp_a
        
        # Update Factors
        # Attack absorbs 70% of error?
        self.team_attack[home] = h_att + (lr * err_h / baseline)
        self.team_defense[away] = a_def + (lr * err_h / baseline) # Conceded more = Defense score goes UP (Bad) or..
        # Wait, defense score: 1.2 means concedes 20% MORE. So higher is worse. 
        # If hg > exp, away defense underestimated -> Increase score. Correct.
        
        self.team_attack[away] = a_att + (lr * err_a / baseline)
        self.team_defense[home] = h_def + (lr * err_a / baseline)
        
        # Clamp to reasonable values
        for t in [home, away]:
            self.team_attack[t] = max(0.2, min(3.0, self.team_attack[t]))
            self.team_defense[t] = max(0.2, min(3.0, self.team_defense[t]))

    def predict_match(self, home_team, away_team, league=None):
        # same logic as before...
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)
        
        if league and league in self.league_baselines:
            baseline = self.league_baselines[league]
        elif home_team in self.team_leagues:
            baseline = self.league_baselines.get(self.team_leagues[home_team], self.league_avg_goals)
        else:
            baseline = self.league_avg_goals
            
        lambda_home = baseline * home_attack * away_defense * (1 + self.home_advantage)
        lambda_away = baseline * away_attack * home_defense
        
        max_goals = 6
        prob_home, prob_draw, prob_away = 0.0, 0.0, 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j: prob_home += p
                elif i == j: prob_draw += p
                else: prob_away += p
                
        return {'prob_home': prob_home, 'prob_draw': prob_draw, 'prob_away': prob_away}
        
    def predict(self, df):
        """
        Predict probabilities for all matches in DataFrame.
        """
        results = []
        for _, match in df.iterrows():
            league = match.get('League', None)
            probs = self.predict_match(match['HomeTeam'], match['AwayTeam'], league=league)
            results.append(probs)
        return pd.DataFrame(results)
        
    def save(self, path):
         with open(path, 'wb') as f:
            # Save dict representation
            d = {
                'team_attack': self.team_attack,
                'team_defense': self.team_defense,
                'league_baselines': self.league_baselines,
                'home_advantage': self.home_advantage
            }
            pickle.dump(d, f)
            
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        obj = cls(home_advantage=d.get('home_advantage', 0.15))
        obj.team_attack = d['team_attack']
        obj.team_defense = d['team_defense']
        obj.league_baselines = d.get('league_baselines', {})
        return obj

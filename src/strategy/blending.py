"""
STAVKI V6 Smart Blending Engine
-------------------------------
Determines the optimal blend between Model Probability and Market Probability
based on the liquidity/efficiency of the competition.

Logic:
- High Liquidity (Elite): Market is sharp. We trust it more (Alpha Low).
- Low Liquidity (Minors): Market is inefficient. We trust Model more (Alpha High).

User "Aggressive" Profile:
- Elite: 30% Model (Alpha 0.3)
- Mid:   50% Model (Alpha 0.5)
- Low:   70% Model (Alpha 0.7)
"""

def get_liquidity_tier(sport_key):
    """
    Classify league into liquidity tiers.
    Returns: 'tier1', 'tier2', 'tier3'
    """
    # TIER 1: Elite Global Competitions (Market is extremely sharp)
    # We want to be careful here.
    TIER_1 = {
        'soccer_epl', 
        'soccer_spain_la_liga', 
        'soccer_uefa_champions_league',
        'soccer_unlimited_champs_league', # If applicable
        'basketball_nba',
        'americanfootball_nfl'
    }

    # TIER 2: Major Domestic Leagues (Market is sharp but beatable)
    TIER_2 = {
        'soccer_italy_serie_a',
        'soccer_germany_bundesliga',
        'soccer_france_ligue_one',
        'soccer_efl_champ',
        'soccer_netherlands_eredivisie',
        'basketball_euroleague'
    }

    if sport_key in TIER_1:
        return 'tier1'
    elif sport_key in TIER_2:
        return 'tier2'
    else:
        return 'tier3'

def get_blending_alpha(sport_key, aggressive=True):
    """
    Get the Model Weight (Alpha) for a given league.
    
    Formula:
    P_Final = Alpha * P_Model + (1 - Alpha) * P_Market
    
    Args:
        sport_key: The league identifier
        aggressive: If True, use the user's requested "Alpha Hunter" profile.
    
    Returns:
        float: Alpha value (0.0 to 1.0)
    """
    tier = get_liquidity_tier(sport_key)
    
    if aggressive:
        # User's "Alpha Hunter" Profile
        if tier == 'tier1':
            return 0.30  # 30% Model, 70% Market
        elif tier == 'tier2':
            return 0.50  # 50% Model, 50% Market
        else:
            return 0.70  # 70% Model, 30% Market (~Legacy behavior)
    else:
        # Conservative / Calibration Profile
        if tier == 'tier1':
            return 0.15
        elif tier == 'tier2':
            return 0.25
        else:
            return 0.40

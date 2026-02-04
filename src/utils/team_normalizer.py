#!/usr/bin/env python3
"""
Team Name Normalizer.

Normalizes team names across different API sources:
- The Odds API
- SportMonks
- Betfair
- Historical data

Usage:
    from src.utils.team_normalizer import TeamNormalizer
    
    normalizer = TeamNormalizer()
    canonical = normalizer.normalize("Man United")  # -> "manchester_united"
"""

import re
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


# Canonical name -> list of aliases
TEAM_ALIASES: Dict[str, List[str]] = {
    # =========================================================================
    # PREMIER LEAGUE
    # =========================================================================
    "arsenal": [
        "Arsenal", "Arsenal FC", "The Arsenal", "ARS", "Gunners"
    ],
    "aston_villa": [
        "Aston Villa", "Villa", "Aston Villa FC", "AVL", "Villans"
    ],
    "bournemouth": [
        "Bournemouth", "AFC Bournemouth", "Bournemouth AFC", "BOU", "Cherries"
    ],
    "brentford": [
        "Brentford", "Brentford FC", "BRE", "Bees"
    ],
    "brighton": [
        "Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion",
        "Brighton Hove Albion", "BHA", "Seagulls"
    ],
    "burnley": [
        "Burnley", "Burnley FC", "BUR", "Clarets"
    ],
    "chelsea": [
        "Chelsea", "Chelsea FC", "CHE", "Blues"
    ],
    "crystal_palace": [
        "Crystal Palace", "C Palace", "C. Palace", "CRY", "Eagles"
    ],
    "everton": [
        "Everton", "Everton FC", "EVE", "Toffees"
    ],
    "fulham": [
        "Fulham", "Fulham FC", "FUL", "Cottagers"
    ],
    "ipswich": [
        "Ipswich", "Ipswich Town", "Ipswich Town FC", "IPS", "Tractor Boys"
    ],
    "leicester": [
        "Leicester", "Leicester City", "Leicester City FC", "LEI", "Foxes"
    ],
    "liverpool": [
        "Liverpool", "Liverpool FC", "LIV", "Reds"
    ],
    "luton": [
        "Luton", "Luton Town", "Luton Town FC", "LUT", "Hatters"
    ],
    "manchester_city": [
        "Manchester City", "Man City", "Man. City", "Manchester C",
        "MCFC", "MCI", "Citizens", "City"
    ],
    "manchester_united": [
        "Manchester United", "Man United", "Man Utd", "Man. United",
        "Manchester U", "MUFC", "MUN", "Red Devils", "United"
    ],
    "newcastle": [
        "Newcastle", "Newcastle United", "Newcastle Utd", "Newcastle U",
        "NUFC", "NEW", "Magpies", "Toon"
    ],
    "nottingham_forest": [
        "Nottingham Forest", "Nottm Forest", "Nott'm Forest", "Notts Forest",
        "Forest", "NFO", "Tricky Trees"
    ],
    "sheffield_united": [
        "Sheffield United", "Sheffield Utd", "Sheffield U", "Sheff United",
        "Sheff Utd", "SHU", "Blades"
    ],
    "southampton": [
        "Southampton", "Southampton FC", "SOU", "Saints"
    ],
    "tottenham": [
        "Tottenham", "Tottenham Hotspur", "Spurs", "Tottenham H",
        "TOT", "THFC", "Lilywhites"
    ],
    "west_ham": [
        "West Ham", "West Ham United", "West Ham Utd", "West Ham U",
        "WHU", "Hammers", "Irons"
    ],
    "wolverhampton": [
        "Wolverhampton", "Wolverhampton Wanderers", "Wolves", "Wolverhampton W",
        "WOL", "Wanderers"
    ],
    
    # =========================================================================
    # CHAMPIONSHIP
    # =========================================================================
    "birmingham": [
        "Birmingham", "Birmingham City", "Birmingham City FC", "BIR", "Blues"
    ],
    "blackburn": [
        "Blackburn", "Blackburn Rovers", "Blackburn R", "BBR"
    ],
    "bristol_city": [
        "Bristol City", "Bristol C", "BRC", "Robins"
    ],
    "cardiff": [
        "Cardiff", "Cardiff City", "Cardiff City FC", "CAR", "Bluebirds"
    ],
    "coventry": [
        "Coventry", "Coventry City", "Coventry City FC", "COV", "Sky Blues"
    ],
    "derby": [
        "Derby", "Derby County", "Derby County FC", "DER", "Rams"
    ],
    "hull": [
        "Hull", "Hull City", "Hull City FC", "HUL", "Tigers"
    ],
    "leeds": [
        "Leeds", "Leeds United", "Leeds Utd", "Leeds U", "LEE", "Whites"
    ],
    "middlesbrough": [
        "Middlesbrough", "Middlesbrough FC", "Boro", "MID"
    ],
    "millwall": [
        "Millwall", "Millwall FC", "MIL", "Lions"
    ],
    "norwich": [
        "Norwich", "Norwich City", "Norwich City FC", "NOR", "Canaries"
    ],
    "plymouth": [
        "Plymouth", "Plymouth Argyle", "Plymouth Argyle FC", "PLY", "Pilgrims"
    ],
    "portsmouth": [
        "Portsmouth", "Portsmouth FC", "Pompey", "POR"
    ],
    "preston": [
        "Preston", "Preston North End", "Preston NE", "PNE"
    ],
    "qpr": [
        "QPR", "Queens Park Rangers", "Queen's Park Rangers", "Queens PR"
    ],
    "sheffield_wednesday": [
        "Sheffield Wednesday", "Sheffield Wed", "Sheff Wednesday", "Sheff Wed",
        "SHW", "Owls"
    ],
    "stoke": [
        "Stoke", "Stoke City", "Stoke City FC", "STO", "Potters"
    ],
    "sunderland": [
        "Sunderland", "Sunderland AFC", "SUN", "Black Cats"
    ],
    "swansea": [
        "Swansea", "Swansea City", "Swansea City FC", "SWA", "Swans", "Jacks"
    ],
    "watford": [
        "Watford", "Watford FC", "WAT", "Hornets"
    ],
    "west_brom": [
        "West Brom", "West Bromwich", "West Bromwich Albion", "WBA", "Baggies"
    ],
    
    # =========================================================================
    # LA LIGA
    # =========================================================================
    "real_madrid": [
        "Real Madrid", "Real Madrid CF", "R Madrid", "RMA", "Los Blancos"
    ],
    "barcelona": [
        "Barcelona", "FC Barcelona", "Barca", "Barça", "BAR", "Blaugrana"
    ],
    "atletico_madrid": [
        "Atletico Madrid", "Atlético Madrid", "Atl Madrid", "Atl. Madrid",
        "ATM", "Colchoneros", "Atleti"
    ],
    "sevilla": [
        "Sevilla", "Sevilla FC", "SEV"
    ],
    "real_sociedad": [
        "Real Sociedad", "Real Sociedad de Fútbol", "La Real", "RSO"
    ],
    "villarreal": [
        "Villarreal", "Villarreal CF", "VIL", "Yellow Submarine"
    ],
    "real_betis": [
        "Real Betis", "Betis", "Real Betis Balompié", "BET"
    ],
    "athletic_bilbao": [
        "Athletic Bilbao", "Athletic Club", "Ath Bilbao", "ATH", "Lions"
    ],
    "valencia": [
        "Valencia", "Valencia CF", "VAL", "Los Che"
    ],
    "getafe": [
        "Getafe", "Getafe CF", "GET"
    ],
    "celta_vigo": [
        "Celta Vigo", "Celta de Vigo", "RC Celta", "CEL"
    ],
    "osasuna": [
        "Osasuna", "CA Osasuna", "OSA"
    ],
    "mallorca": [
        "Mallorca", "RCD Mallorca", "Real Mallorca", "MAL"
    ],
    "las_palmas": [
        "Las Palmas", "UD Las Palmas", "LAS"
    ],
    "rayo_vallecano": [
        "Rayo Vallecano", "Rayo", "RAY"
    ],
    "alaves": [
        "Alavés", "Alaves", "Deportivo Alavés", "ALA"
    ],
    "girona": [
        "Girona", "Girona FC", "GIR"
    ],
    "cadiz": [
        "Cádiz", "Cadiz", "Cádiz CF", "CAD"
    ],
    "almeria": [
        "Almería", "Almeria", "UD Almería", "ALM"
    ],
    "granada": [
        "Granada", "Granada CF", "GRA"
    ],
    
    # =========================================================================
    # SERIE A
    # =========================================================================
    "ac_milan": [
        "AC Milan", "Milan", "A.C. Milan", "MIL", "Rossoneri"
    ],
    "inter": [
        "Inter", "Inter Milan", "Internazionale", "Inter Milano",
        "FC Internazionale", "INT", "Nerazzurri"
    ],
    "juventus": [
        "Juventus", "Juventus FC", "Juve", "JUV", "Old Lady", "Bianconeri"
    ],
    "napoli": [
        "Napoli", "SSC Napoli", "NAP", "Partenopei"
    ],
    "roma": [
        "Roma", "AS Roma", "A.S. Roma", "ROM", "Giallorossi"
    ],
    "lazio": [
        "Lazio", "SS Lazio", "LAZ", "Biancocelesti"
    ],
    "atalanta": [
        "Atalanta", "Atalanta BC", "Atalanta Bergamo", "ATA", "La Dea"
    ],
    "fiorentina": [
        "Fiorentina", "ACF Fiorentina", "Viola", "FIO"
    ],
    "torino": [
        "Torino", "Torino FC", "TOR", "Granata"
    ],
    "bologna": [
        "Bologna", "Bologna FC", "Bologna 1909", "BOL"
    ],
    "monza": [
        "Monza", "AC Monza", "MON"
    ],
    "udinese": [
        "Udinese", "Udinese Calcio", "UDI"
    ],
    "sassuolo": [
        "Sassuolo", "US Sassuolo", "SAS"
    ],
    "empoli": [
        "Empoli", "Empoli FC", "EMP"
    ],
    "salernitana": [
        "Salernitana", "US Salernitana", "SAL"
    ],
    "cagliari": [
        "Cagliari", "Cagliari Calcio", "CAG"
    ],
    "verona": [
        "Verona", "Hellas Verona", "HEL"
    ],
    "lecce": [
        "Lecce", "US Lecce", "LEC"
    ],
    "frosinone": [
        "Frosinone", "Frosinone Calcio", "FRO"
    ],
    "genoa": [
        "Genoa", "Genoa CFC", "GEN"
    ],
    
    # =========================================================================
    # BUNDESLIGA
    # =========================================================================
    "bayern_munich": [
        "Bayern Munich", "Bayern München", "Bayern", "FC Bayern",
        "FC Bayern München", "BAY", "Bavarians"
    ],
    "borussia_dortmund": [
        "Borussia Dortmund", "Dortmund", "BVB", "Borussia D", "DOR"
    ],
    "rb_leipzig": [
        "RB Leipzig", "Leipzig", "RasenBallsport Leipzig", "RBL", "LEI"
    ],
    "bayer_leverkusen": [
        "Bayer Leverkusen", "Leverkusen", "Bayer 04", "B04", "LEV"
    ],
    "eintracht_frankfurt": [
        "Eintracht Frankfurt", "Frankfurt", "E Frankfurt", "SGE", "FRA"
    ],
    "union_berlin": [
        "Union Berlin", "1. FC Union Berlin", "Union B", "FCU", "UNB"
    ],
    "freiburg": [
        "Freiburg", "SC Freiburg", "FRE"
    ],
    "wolfsburg": [
        "Wolfsburg", "VfL Wolfsburg", "WOB"
    ],
    "hoffenheim": [
        "Hoffenheim", "TSG Hoffenheim", "TSG 1899", "HOF"
    ],
    "mainz": [
        "Mainz", "1. FSV Mainz 05", "Mainz 05", "MAI", "M05"
    ],
    "borussia_monchengladbach": [
        "Borussia Mönchengladbach", "Borussia Monchengladbach", "Gladbach",
        "Mönchengladbach", "Monchengladbach", "BMG", "MON"
    ],
    "werder_bremen": [
        "Werder Bremen", "Bremen", "SV Werder", "SVW", "BRE"
    ],
    "stuttgart": [
        "Stuttgart", "VfB Stuttgart", "VfB", "STU"
    ],
    "augsburg": [
        "Augsburg", "FC Augsburg", "FCA", "AUG"
    ],
    "koln": [
        "Köln", "Koln", "1. FC Köln", "1. FC Koln", "Cologne", "KOE"
    ],
    "heidenheim": [
        "Heidenheim", "1. FC Heidenheim", "HDH"
    ],
    "darmstadt": [
        "Darmstadt", "SV Darmstadt 98", "Darmstadt 98", "DAR"
    ],
    "bochum": [
        "Bochum", "VfL Bochum", "BOC"
    ],
    
    # =========================================================================
    # LIGUE 1
    # =========================================================================
    "psg": [
        "PSG", "Paris Saint-Germain", "Paris Saint Germain", "Paris SG",
        "Paris", "PAR"
    ],
    "marseille": [
        "Marseille", "Olympique Marseille", "Olympique de Marseille",
        "OM", "MAR"
    ],
    "monaco": [
        "Monaco", "AS Monaco", "ASM", "MON"
    ],
    "lyon": [
        "Lyon", "Olympique Lyon", "Olympique Lyonnais", "OL", "LYO"
    ],
    "lille": [
        "Lille", "LOSC", "LOSC Lille", "LIL"
    ],
    "lens": [
        "Lens", "RC Lens", "Racing Lens", "LEN"
    ],
    "nice": [
        "Nice", "OGC Nice", "NIC"
    ],
    "rennes": [
        "Rennes", "Stade Rennais", "Stade Rennais FC", "REN"
    ],
    "reims": [
        "Reims", "Stade de Reims", "REI"
    ],
    "montpellier": [
        "Montpellier", "Montpellier HSC", "MON", "MHP"
    ],
    "toulouse": [
        "Toulouse", "Toulouse FC", "TOU", "TFC"
    ],
    "nantes": [
        "Nantes", "FC Nantes", "NAN"
    ],
    "strasbourg": [
        "Strasbourg", "Racing Strasbourg", "RC Strasbourg", "STR"
    ],
    "brest": [
        "Brest", "Stade Brestois", "Stade Brestois 29", "BRE"
    ],
    "le_havre": [
        "Le Havre", "Le Havre AC", "HAV"
    ],
    "lorient": [
        "Lorient", "FC Lorient", "LOR"
    ],
    "metz": [
        "Metz", "FC Metz", "MET"
    ],
    "clermont": [
        "Clermont", "Clermont Foot", "Clermont Foot 63", "CLE"
    ],
}


# SportMonks team ID mapping (partial - extend as needed)
SPORTMONKS_IDS: Dict[str, int] = {
    "arsenal": 19,
    "aston_villa": 66,
    "bournemouth": 127,
    "brentford": 108,
    "brighton": 47,
    "burnley": 48,
    "chelsea": 18,
    "crystal_palace": 62,
    "everton": 24,
    "fulham": 63,
    "ipswich": 161,
    "leicester": 46,
    "liverpool": 8,
    "luton": 96,
    "manchester_city": 9,
    "manchester_united": 14,
    "newcastle": 23,
    "nottingham_forest": 64,
    "sheffield_united": 51,
    "southampton": 7,
    "tottenham": 6,
    "west_ham": 29,
    "wolverhampton": 33,
    # Add more as needed...
}


class TeamNormalizer:
    """
    Normalize team names across different data sources.
    
    Usage:
        normalizer = TeamNormalizer()
        canonical = normalizer.normalize("Man United")  # -> "manchester_united"
        sportmonks_id = normalizer.get_sportmonks_id("Man United")  # -> 14
    """
    
    def __init__(self):
        # Build reverse lookup
        self._alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in TEAM_ALIASES.items():
            # Add canonical name itself
            self._alias_to_canonical[canonical.lower()] = canonical
            self._alias_to_canonical[self._clean(canonical)] = canonical
            
            # Add all aliases
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical
                self._alias_to_canonical[self._clean(alias)] = canonical
        
        logger.info(f"TeamNormalizer initialized with {len(TEAM_ALIASES)} teams, "
                   f"{len(self._alias_to_canonical)} aliases")
    
    def _clean(self, name: str) -> str:
        """Clean team name for matching."""
        # Remove special chars, lowercase
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
        # Remove common suffixes
        clean = re.sub(r'\s+(fc|afc|sc|cf|ssc)$', '', clean)
        return clean.strip()
    
    def normalize(self, name: str) -> str:
        """
        Normalize team name to canonical form.
        
        Args:
            name: Raw team name from any source
            
        Returns:
            Canonical team name (lowercase, underscores)
        """
        if not name:
            return ""
        
        # Direct lookup
        lower = name.lower()
        if lower in self._alias_to_canonical:
            return self._alias_to_canonical[lower]
        
        # Clean and lookup
        clean = self._clean(name)
        if clean in self._alias_to_canonical:
            return self._alias_to_canonical[clean]
        
        # Fuzzy match
        best_match = self._fuzzy_match(name)
        if best_match:
            logger.debug(f"Fuzzy matched '{name}' -> '{best_match}'")
            return best_match
        
        # Return cleaned version as fallback
        logger.warning(f"Unknown team: {name}")
        return self._clean(name).replace(' ', '_')
    
    def _fuzzy_match(self, name: str, threshold: float = 0.8) -> Optional[str]:
        """Find best fuzzy match for a team name."""
        clean = self._clean(name)
        best_score = 0.0
        best_match = None
        
        for alias, canonical in self._alias_to_canonical.items():
            score = SequenceMatcher(None, clean, alias).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = canonical
        
        return best_match
    
    def get_sportmonks_id(self, name: str) -> Optional[int]:
        """Get SportMonks team ID for a team name."""
        canonical = self.normalize(name)
        return SPORTMONKS_IDS.get(canonical)
    
    def get_all_aliases(self, name: str) -> List[str]:
        """Get all known aliases for a team."""
        canonical = self.normalize(name)
        return TEAM_ALIASES.get(canonical, [canonical])
    
    def add_alias(self, canonical: str, alias: str):
        """Add a new alias for a team."""
        if canonical not in TEAM_ALIASES:
            TEAM_ALIASES[canonical] = []
        
        TEAM_ALIASES[canonical].append(alias)
        self._alias_to_canonical[alias.lower()] = canonical
        self._alias_to_canonical[self._clean(alias)] = canonical
    
    def is_same_team(self, name1: str, name2: str) -> bool:
        """Check if two names refer to the same team."""
        return self.normalize(name1) == self.normalize(name2)


# Singleton instance
_normalizer: Optional[TeamNormalizer] = None


def get_normalizer() -> TeamNormalizer:
    """Get singleton normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = TeamNormalizer()
    return _normalizer


def normalize_team(name: str) -> str:
    """Convenience function to normalize a team name."""
    return get_normalizer().normalize(name)


def get_sportmonks_id(name: str) -> Optional[int]:
    """Convenience function to get SportMonks ID."""
    return get_normalizer().get_sportmonks_id(name)


# CLI for testing
if __name__ == "__main__":
    import sys
    
    normalizer = TeamNormalizer()
    
    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        print(f"Input: {name}")
        print(f"Canonical: {normalizer.normalize(name)}")
        print(f"SportMonks ID: {normalizer.get_sportmonks_id(name)}")
        print(f"Aliases: {normalizer.get_all_aliases(name)}")
    else:
        # Run tests
        test_cases = [
            ("Man United", "manchester_united"),
            ("Manchester United", "manchester_united"),
            ("MUFC", "manchester_united"),
            ("Man City", "manchester_city"),
            ("Bayern Munich", "bayern_munich"),
            ("Bayern München", "bayern_munich"),
            ("Real Madrid CF", "real_madrid"),
            ("PSG", "psg"),
            ("Inter Milan", "inter"),
            ("AC Milan", "ac_milan"),
        ]
        
        print("Team Name Normalizer Tests")
        print("=" * 50)
        
        passed = 0
        for input_name, expected in test_cases:
            result = normalizer.normalize(input_name)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            print(f"{status} '{input_name}' -> '{result}' (expected: '{expected}')")
        
        print("=" * 50)
        print(f"Passed: {passed}/{len(test_cases)}")

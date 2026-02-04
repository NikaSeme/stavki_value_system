# 🔌 API Integration Master Plan
## SportMonks + Betfair + OpenWeatherMap + The Odds API

---

# 🎯 Что у нас есть

```mermaid
flowchart TB
    subgraph APIS["🌐 Наши API"]
        SM["🏆 SportMonks<br/>European Advanced<br/>💎 PREMIUM"]
        BF["📊 Betfair<br/>Exchange<br/>💰 FREE"]
        OW["🌤️ OpenWeatherMap<br/>💰 FREE"]
        OA["📈 The Odds API<br/>💰 FREE TIER"]
    end
    
    subgraph DATA["📥 Данные"]
        D1["xG, статистика"]
        D2["Составы, травмы"]
        D3["Odds multi-source"]
        D4["Погода"]
        D5["True market odds"]
    end
    
    SM --> D1
    SM --> D2
    SM --> D3
    OA --> D3
    BF --> D5
    OW --> D4
```

---

# 📊 Разбор каждого API

````carousel
## 🏆 SportMonks European Advanced

**Это наш ГЛАВНЫЙ источник!**

| Возможность | Что даёт модели |
|:---|:---|
| **xG Data** | Expected goals — лучше чем actual goals |
| **99+ статистик** | Shots, passes, possession, etc. |
| **Lineups** | Составы перед матчем |
| **Injuries** | Травмы и дисквалификации |
| **Odds** | Pre-match и in-play |
| **Predictions** | Их прогнозы (для сравнения) |
| **Transfers** | Трансферы и слухи |
| **Weather forecast** | Прогноз погоды! |

**27 европейских лиг включая:**
- Premier League, La Liga, Bundesliga
- Serie A, Ligue 1, Championship
- И ещё 20+ лиг
<!-- slide -->
## 📊 Betfair Exchange

**Для чего использовать:**

| Функция | Польза |
|:---|:---|
| **True odds** | Без маржи букмекера |
| **Historical data** | Для backtesting |
| **Volume/Liquidity** | Сколько денег на рынке |
| **Sharp line** | Биржа = истина рынка |

**CLV Benchmark:**
- Betfair closing = "правда"
- Если наши ставки бьют closing → мы прибыльны
<!-- slide -->
## 🌤️ OpenWeatherMap

**Для чего использовать:**

| Данные | Влияние на матч |
|:---|:---|
| **Дождь** | Меньше голов, больше ничьих |
| **Ветер** | Меньше точных передач |
| **Температура** | Экстремальная → усталость |
| **Влажность** | Влияет на темп игры |

**Но!** SportMonks УЖЕ включает weather forecast!
→ OpenWeatherMap = backup/validation
<!-- slide -->
## 📈 The Odds API

**Для чего использовать:**

| Функция | Польза |
|:---|:---|
| **10+ букмекеров** | Best price hunting |
| **Pinnacle odds** | Sharp line benchmark |
| **Fast updates** | Быстрее чем SportMonks |

**Стратегия:**
- The Odds API = основной для odds
- SportMonks odds = backup/validation
````

---

# 🎯 Стратегия использования

```mermaid
flowchart TD
    subgraph STATS["📊 СТАТИСТИКА"]
        SM1["SportMonks"] --> |"xG, shots, passes"| FEATURES
    end
    
    subgraph SQUAD["👥 СОСТАВЫ"]
        SM2["SportMonks"] --> |"lineups, injuries"| FEATURES
    end
    
    subgraph ODDS["💹 КОЭФФИЦИЕНТЫ"]
        OA["The Odds API"] --> |"multi-bookmaker"| BEST["Best Price<br/>Selector"]
        SM3["SportMonks"] --> |"validation"| BEST
        BF["Betfair"] --> |"CLV benchmark"| BEST
    end
    
    subgraph WEATHER["🌤️ ПОГОДА"]
        SM4["SportMonks"] --> |"primary"| FEATURES
        OW["OpenWeatherMap"] --> |"backup"| FEATURES
    end
    
    FEATURES["🔧 Feature<br/>Engine"]
    BEST --> FEATURES
    FEATURES --> MODEL["🤖 ML Model"]
```

---

# 📋 Матрица: Что откуда брать

| Данные | Основной источник | Backup | Зачем backup |
|:---|:---|:---|:---|
| **xG, статистика** | SportMonks | — | Единственный источник |
| **Lineups** | SportMonks | — | Единственный источник |
| **Injuries** | SportMonks | — | Единственный источник |
| **Odds (betting)** | The Odds API | SportMonks | Больше букмекеров |
| **CLV benchmark** | Betfair | Pinnacle via Odds API | True market |
| **Weather** | SportMonks | OpenWeatherMap | Validation |
| **Historical odds** | Betfair | The Odds API snapshots | Backtesting |

---

# 🔧 Новые признаки (Features)

## Из SportMonks (+30 features):

```mermaid
mindmap
  root((SportMonks<br/>Features))
    xG
      home_xg_season
      away_xg_season
      home_xga_season
      away_xga_season
      xg_overperformance
    Stats
      shots_on_target_avg
      possession_avg
      passes_accuracy
      corners_avg
      fouls_avg
    Squad
      injuries_count
      suspensions_count
      avg_squad_age
      squad_market_value
      key_player_missing
    Form
      last5_xg
      last5_xga
      goals_vs_xg_diff
```

## Из Betfair (+5 features):

| Feature | Описание |
|:---|:---|
| `betfair_home_odds` | True market odds |
| `betfair_volume_home` | Liquidity |
| `odds_vs_betfair_diff` | Gap от истины рынка |
| `betfair_movement_24h` | Движение за 24ч |
| `market_confidence` | Volume / total volume |

## Из Weather (+5 features):

| Feature | Описание |
|:---|:---|
| `temperature` | Celsius |
| `precipitation` | mm/hour |
| `wind_speed` | km/h |
| `humidity` | % |
| `weather_score` | Composite 0-1 |

---

# 📅 План реализации

```mermaid
gantt
    title API Integration Timeline
    dateFormat  YYYY-MM-DD
    
    section Phase 1: Core
    SportMonks client       :p1a, 2026-02-05, 2d
    xG features extraction  :p1b, after p1a, 2d
    Lineup/Injuries         :p1c, after p1b, 1d
    
    section Phase 2: Odds
    Betfair client          :p2a, 2026-02-10, 2d
    CLV tracking            :p2b, after p2a, 2d
    Best price selector     :p2c, after p2b, 1d
    
    section Phase 3: Weather
    Weather integration     :p3a, 2026-02-15, 1d
    
    section Phase 4: ML
    Feature pipeline update :p4a, 2026-02-16, 2d
    Model retraining        :p4b, after p4a, 2d
    Testing & validation    :p4c, after p4b, 2d
```

---

# 👥 Разделение работ

## 👤 Что нужно ОТ ТЕБЯ:

| Задача | Время | Когда |
|:---|:---:|:---|
| **SportMonks API key** | 1 мин | Сейчас |
| **Betfair credentials** | 5 мин | Сейчас |
| **OpenWeatherMap key** | 2 мин | Сейчас |
| **Подтвердить план** | 5 мин | После прочтения |

---

## 🤖 Что сделаю Я:

### Phase 1: SportMonks Integration (3-4 дня)
- [ ] `SportMonksClient` — API клиент
- [ ] `XGFeatureExtractor` — извлечение xG данных
- [ ] `LineupTracker` — отслеживание составов
- [ ] `InjuryMonitor` — мониторинг травм
- [ ] Тесты для всего

### Phase 2: Odds Integration (3-4 дня)
- [ ] `BetfairClient` — клиент биржи
- [ ] `CLVTracker` — отслеживание CLV
- [ ] `BestPriceSelector` — поиск лучшей цены
- [ ] Интеграция с существующим The Odds API

### Phase 3: Weather (1 день)
- [ ] `WeatherEnricher` — добавление погоды
- [ ] Fallback на OpenWeatherMap

### Phase 4: ML Pipeline Update (4 дня)
- [ ] Обновление `LiveFeatureExtractor` (50+ features)
- [ ] Ретрейнинг моделей на новых данных
- [ ] A/B тест старые vs новые features
- [ ] Валидация улучшения ROI

---

# 💰 Ожидаемый Impact

| Источник | Новые features | Ожидаемый ROI boost |
|:---|:---:|:---:|
| **SportMonks xG** | +10 | **+2-3%** |
| **SportMonks lineups** | +5 | **+1-2%** |
| **Betfair CLV** | +5 | **+1-2%** |
| **Weather** | +5 | **+0.5-1%** |
| **Best price** | — | **+1-2%** |
| **ИТОГО** | **+25 features** | **+5.5-10% ROI** |

---

# ⏭️ Следующий шаг

> Скинь мне API ключи, и я начну реализацию Phase 1!

**Нужно:**
1. 🏆 **SportMonks API Key**
2. 📊 **Betfair App Key + Session Token**
3. 🌤️ **OpenWeatherMap API Key**

---

# ❓ Вопросы перед стартом

1. **Частота обновления данных?**
   - [ ] Каждый час (экономит API calls)
   - [ ] Каждые 30 мин (баланс)
   - [ ] Каждые 15 мин (максимум)

2. **Хранение данных?**
   - [ ] SQLite (просто, уже есть)
   - [ ] PostgreSQL (надёжнее)
   - [ ] TimescaleDB (для time-series)

3. **Приоритет лиг?**
   - [ ] Только Big 5 (EPL, La Liga, Bundesliga, Serie A, Ligue 1)
   - [ ] Big 5 + Championship
   - [ ] Все 27 европейских лиг

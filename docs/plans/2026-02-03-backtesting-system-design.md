# 🧪 Ultimate Backtesting System — Design Document

> **Цель:** Создать максимально эффективную систему тестирования для калибровки всех компонентов STAVKI

---

## 🎯 Три подхода

### Подход A: **Реалистичный** (рекомендую как базу)

**Что включает:**
- Walk-Forward Optimization
- Monte Carlo симуляции
- Per-league калибровка
- CLV (Closing Line Value) tracking

**Плюсы:** Быстро реализовать, покрывает 80% потребностей
**Минусы:** Не учитывает некоторые рыночные эффекты

---

### Подход B: **Продвинутый**

**Всё из A, плюс:**
- Multi-bookmaker arbitrage detection
- Liquidity modeling (можно ли исполнить ставку?)
- Slippage simulation (коэфф изменится пока ставим)
- Correlation analysis между лигами

**Плюсы:** Намного ближе к реальности
**Минусы:** Нужны дополнительные данные

---

### Подход C: **Максималистский** (всё что возможно)

**Всё из A + B, плюс:**
- Reinforcement Learning для динамической стратегии
- Regime detection (разные рынки = разные стратегии)
- Black Swan simulation
- Real-time odds streaming backtest
- Paper trading mode с реальными API

**Плюсы:** Cutting-edge, конкурентное преимущество
**Минусы:** Сложнее, дольше

---

## ✅ Рекомендация: Подход C (Максималистский)

Раз бюджет и сложность не ограничены — берём всё!

---

# 📐 Дизайн системы

## Раздел 1: Архитектура

```mermaid
flowchart TB
    subgraph DATA["📥 DATA LAYER"]
        D1["Historical Odds<br/>football-data.co.uk"]
        D2["Live Odds Snapshots<br/>The Odds API"]
        D3["Results Database"]
        D4["Features Store"]
    end
    
    subgraph ENGINE["⚙️ BACKTEST ENGINE"]
        E1["Walk-Forward<br/>Optimizer"]
        E2["Monte Carlo<br/>Simulator"]
        E3["Reality<br/>Simulator"]
        E4["Stress<br/>Tester"]
    end
    
    subgraph CALIBRATION["🎯 CALIBRATION"]
        C1["Model Weights"]
        C2["Kelly Fractions"]
        C3["EV Thresholds"]
        C4["League-specific<br/>Parameters"]
    end
    
    subgraph OUTPUT["📊 OUTPUT"]
        O1["Metrics Dashboard"]
        O2["Confidence Intervals"]
        O3["Recommendations"]
        O4["Risk Reports"]
    end
    
    DATA --> ENGINE --> CALIBRATION --> OUTPUT
```

---

## Раздел 2: Модули системы

### 2.1 🔄 Walk-Forward Optimization

**Что это:** Тренируем на прошлом, тестируем на будущем, двигаемся вперёд по времени.

```mermaid
gantt
    title Walk-Forward Timeline
    dateFormat  YYYY-MM
    section Fold 1
    Train     :2021-01, 6M
    Test      :2021-07, 2M
    section Fold 2
    Train     :2021-03, 6M
    Test      :2021-09, 2M
    section Fold 3
    Train     :2021-05, 6M
    Test      :2021-11, 2M
```

**Зачем:** Предотвращает overfitting — если модель работает на всех fold'ах, она реально работает.

**Параметры для оптимизации:**
| Параметр | Диапазон | Шаг |
|----------|----------|-----|
| `ensemble_weight_poisson` | 0.0 - 1.0 | 0.05 |
| `ensemble_weight_catboost` | 0.0 - 1.0 | 0.05 |
| `ensemble_weight_neural` | 0.0 - 1.0 | 0.05 |
| `kelly_fraction` | 0.1 - 0.5 | 0.05 |
| `min_ev_threshold` | 0.03 - 0.15 | 0.01 |
| `min_odds` | 1.3 - 2.0 | 0.1 |

---

### 2.2 🎲 Monte Carlo Simulator

**Что это:** Прогоняем 10,000+ случайных сценариев чтобы понять распределение результатов.

```mermaid
flowchart LR
    subgraph MC["Monte Carlo Engine"]
        R["Random<br/>Resampling"]
        V["Variance<br/>Injection"]
        B["Bootstrap<br/>Samples"]
    end
    
    H["Historical<br/>Bets"] --> MC
    MC --> D["Distribution<br/>of Outcomes"]
    D --> CI["95% Confidence<br/>Interval"]
    D --> VaR["Value at Risk"]
    D --> ES["Expected<br/>Shortfall"]
```

**Что получаем:**
- **95% CI для ROI:** "ROI будет между 3% и 12% с 95% уверенностью"
- **Value at Risk (VaR):** "С 5% вероятностью потеряем больше X"
- **Maximum Drawdown distribution:** "Типичный drawdown 15-25%"

---

### 2.3 ⚡ Reality Simulator

**Что это:** Симуляция реальных условий рынка.

```mermaid
flowchart TB
    subgraph REALITY["Реальные факторы"]
        F1["⏱️ Latency<br/>50-200ms delay"]
        F2["📉 Slippage<br/>Odds change -1%"]
        F3["💧 Liquidity<br/>Max €500-5000"]
        F4["🚫 Limits<br/>Bookmaker bans"]
        F5["📊 Line Movement<br/>Closing line value"]
    end
    
    B["Backtest Bet"] --> REALITY
    REALITY --> A["Adjusted<br/>Outcome"]
```

**Сценарии:**
| Сценарий | Что симулируем |
|----------|----------------|
| **Optimistic** | Всё идеально |
| **Realistic** | 1-2% slippage, 100ms delay |
| **Pessimistic** | 5% slippage, limits после 50 ставок |
| **Worst Case** | 10% slippage, быстрые лимиты |

---

### 2.4 🌪️ Stress Tester

**Что это:** Проверка системы в экстремальных условиях.

```mermaid
flowchart LR
    subgraph STRESS["Stress Scenarios"]
        S1["📉 10 проигрышей<br/>подряд"]
        S2["🦢 Black Swan<br/>COVID-2020"]
        S3["📊 Низкая волатильность<br/>рынка"]
        S4["⚡ Spike в odds<br/>anomalies"]
    end
    
    STRESS --> T["Тест<br/>стратегии"]
    T --> R["Отчёт по<br/>устойчивости"]
```

**Black Swan симуляции:**
- **COVID scenario:** 3 месяца без футбола, потом резкий рестарт
- **Fixing scandal:** Внезапная аномалия в одной лиге
- **Bookmaker failure:** Один крупный букмекер закрывается
- **Model degradation:** Модель начинает ошибаться (drift detection)

---

### 2.5 🎓 AutoML Calibrator

**Что это:** Автоматический поиск оптимальных параметров для каждой лиги.

```mermaid
flowchart TB
    subgraph SEARCH["Hyperparameter Search"]
        B["Bayesian<br/>Optimization"]
        G["Grid<br/>Search"]
        R["Random<br/>Search"]
    end
    
    subgraph LEAGUES["Per-League"]
        L1["EPL"]
        L2["La Liga"]
        L3["Bundesliga"]
        L4["Serie A"]
    end
    
    SEARCH --> LEAGUES
    LEAGUES --> O["Optimal<br/>Parameters<br/>per League"]
```

**Отдельные параметры для каждой лиги:**
```json
{
  "EPL": {
    "poisson_weight": 0.35,
    "catboost_weight": 0.40,
    "neural_weight": 0.25,
    "kelly": 0.20,
    "min_ev": 0.06
  },
  "Bundesliga": {
    "poisson_weight": 0.45,
    "catboost_weight": 0.35,
    "neural_weight": 0.20,
    "kelly": 0.25,
    "min_ev": 0.05
  }
}
```

---

### 2.6 📈 CLV Tracker (Closing Line Value)

**Что это:** Сравнение наших коэффициентов с закрывающей линией.

```mermaid
flowchart LR
    subgraph CLV["CLV Analysis"]
        O1["Odds at bet<br/>2.40"]
        O2["Closing odds<br/>2.25"]
        C["CLV = +6.7%"]
    end
    
    O1 --> C
    O2 --> C
    C --> V["✅ Beating<br/>the market"]
```

**Зачем:** CLV — лучший индикатор долгосрочного edge. Если постоянно бьём closing line — мы прибыльны.

**Метрики:**
- **CLV%:** Средний % выигрыша у закрывающей линии
- **CLV Hit Rate:** % ставок с положительным CLV
- **CLV by League:** CLV разбитый по лигам

---

### 2.7 🤖 Regime Detector

**Что это:** Определение "режима" рынка для адаптации стратегии.

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> HighVolatility: Market shock
    Normal --> LowVolatility: Stable period
    HighVolatility --> Normal: Stabilization
    LowVolatility --> Normal: Event trigger
    
    Normal: Standard strategy
    HighVolatility: Reduce stakes
    LowVolatility: Increase thresholds
```

**Режимы:**
| Режим | Характеристики | Действие |
|-------|----------------|----------|
| **Normal** | Обычная волатильность | Стандартная стратегия |
| **High Volatility** | Много движения линий | Уменьшить Kelly |
| **Low Edge** | Рынок эффективен | Повысить EV threshold |
| **Opportunity** | Много value | Увеличить exposure |

---

## Раздел 3: Метрики и Dashboard

### Основные метрики

| Метрика | Описание | Цель |
|---------|----------|------|
| **ROI** | Return on Investment | > 5% |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Max Drawdown** | Максимальная просадка | < 25% |
| **Win Rate** | % выигранных ставок | > 40% |
| **CLV** | Closing Line Value | > 2% |
| **Kelly Efficiency** | Реальный vs теоретический Kelly | > 80% |

### Продвинутые метрики

| Метрика | Описание |
|---------|----------|
| **Calmar Ratio** | ROI / Max Drawdown |
| **Sortino Ratio** | Return vs downside deviation |
| **Recovery Factor** | Profit / Max Drawdown |
| **Profit Factor** | Gross profit / Gross loss |
| **Expected Shortfall** | Average loss in worst 5% scenarios |

---

## Раздел 4: Данные и инфраструктура

### 4.1 Источники данных

```mermaid
flowchart LR
    subgraph FREE["Бесплатные"]
        F1["football-data.co.uk<br/>История 20+ лет"]
        F2["The Odds API<br/>500 req/month free"]
    end
    
    subgraph PAID["Платные (опционально)"]
        P1["Betfair Exchange<br/>Real liquidity"]
        P2["Pinnacle API<br/>Sharp line"]
        P3["Sportradar<br/>Real-time odds"]
    end
    
    subgraph OWN["Своё"]
        O1["Odds Snapshots<br/>Накапливаем"]
        O2["Bet History<br/>Наши ставки"]
    end
```

### 4.2 Snapshot Collection System

**Идея:** Каждый час сохраняем коэффициенты → через год у нас миллионы точек данных.

```python
# Структура снапшота
{
    "timestamp": "2024-01-15T14:00:00Z",
    "event_id": "epl_manu_liv_2024",
    "bookmakers": {
        "bet365": {"home": 2.40, "draw": 3.20, "away": 3.00},
        "pinnacle": {"home": 2.38, "draw": 3.25, "away": 2.98},
        ...
    },
    "time_to_kickoff_hours": 24
}
```

---

## Раздел 5: Plan реализации

### Phase 1: Foundation (1-2 недели)

- [ ] **BacktestEngine class** — Базовый движок
- [ ] **DataLoader** — Унифицированная загрузка данных
- [ ] **MetricsCalculator** — Все метрики в одном месте
- [ ] **Walk-Forward Validator** — Базовая версия

### Phase 2: Simulation (2-3 недели)

- [ ] **MonteCarloSimulator** — 10K+ симуляций
- [ ] **RealitySimulator** — Slippage, latency, limits
- [ ] **StressTester** — Black swan scenarios
- [ ] **CLVTracker** — Closing line analysis

### Phase 3: Optimization (2-3 недели)

- [ ] **AutoCalibrator** — Bayesian optimization
- [ ] **PerLeagueOptimizer** — Отдельные параметры для лиг
- [ ] **RegimeDetector** — Market regime classification
- [ ] **EnsembleWeightOptimizer** — Оптимизация весов моделей

### Phase 4: Dashboard & Integration (1-2 недели)

- [ ] **MetricsDashboard** — Визуализация результатов
- [ ] **ConfidenceReporter** — Confidence intervals
- [ ] **Integration with production** — Auto-update parameters
- [ ] **Paper Trading Mode** — Тест на реальных данных без денег

---

## 📊 Ожидаемые результаты

После внедрения системы мы получим:

| Возможность | Польза |
|-------------|--------|
| **Оптимальные веса по лигам** | +2-5% ROI |
| **Правильный Kelly** | Меньше drawdown |
| **CLV tracking** | Proof of edge |
| **Stress testing** | Готовность к Black Swan |
| **Confidence intervals** | Знаем реальные риски |
| **Auto-recalibration** | Адаптация к изменениям рынка |

---

## ⚠️ Вопросы для обсуждения

1. **Приоритет фаз** — Начинаем с Phase 1 последовательно или параллелим?

2. **Платные данные** — Готов ли инвестировать в Pinnacle/Betfair API для более точных данных?

3. **Paper Trading** — Хочешь режим "виртуальных ставок" для тестирования в реальном времени?

4. **Dashboard** — Нужен ли веб-интерфейс или достаточно CLI + JSON отчётов?

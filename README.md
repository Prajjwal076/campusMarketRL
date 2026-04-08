---
title: Campus Market Environment
sdk: docker
app_port: 7860
---

<div align="center">

# Campus Market Environment

### Reinforcement Learning Environment for Campus Shop Decision-Making

**One shop. Many market forces. Long-horizon decisions.**

![Built with FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)
![RL Environment](https://img.shields.io/badge/Type-RL%20Environment-blue)
![Docker Ready](https://img.shields.io/badge/Deploy-Docker-2496ED)
![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)

</div>

---

## What Is This?

`campus_market_env` is a reinforcement learning environment that simulates a campus shop operating under changing customer demand, competitor pressure, seasonality, inventory dynamics, budget constraints, and random market shocks.

An agent controls one shop by choosing:

- `price_adjustment`
- `marketing_spend`
- `restock_amount`
- `product_focus`

On every step, the environment updates:

- traffic
- conversion
- revenue
- customer satisfaction
- inventory level
- awareness
- market sentiment
- competitor pressure
- monthly budget

The environment is exposed through a FastAPI API and can be used locally through Python code, HTTP requests, or the included client.

---

## Why It's Useful

This environment is designed for testing agents that need to balance:

- short-term profit vs long-term health
- pricing vs conversion
- marketing vs budget efficiency
- restocking vs overstock risk
- satisfaction vs operational discipline
- deterministic planning vs random shocks

It works well for:

- RL experiments
- policy evaluation
- environment prototyping
- LLM-agent decision testing
- reward function design and ablation studies

---

## Quick Start

### Run with Docker

```bash
docker build -t campus-market .
docker run -p 7860:7860 campus-market
```

Then open:

- `http://localhost:7860/`
- `http://localhost:7860/docs`

Health check:

```bash
curl http://localhost:7860/api/health
```

### Run with Python

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
python main.py
```

### Local Smoke Test

```bash
python test_env.py
```

---

## Example API Usage

### Reset

```bash
curl -X POST http://localhost:7860/api/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 7}'
```

### Step

```bash
curl -X POST http://localhost:7860/api/step \
  -H "Content-Type: application/json" \
  -d '{
    "price_adjustment": 0.1,
    "marketing_spend": 100.0,
    "restock_amount": 10,
    "product_focus": "food"
  }'
```

---

## Environment Overview

### Core Loop

Each environment step follows this flow:

1. The agent submits a market action.
2. The engine generates trend, student demand clusters, and competitors.
3. The simulator computes awareness, traffic, conversion, revenue, inventory, and satisfaction.
4. Random events may alter price pressure, inventory, or competition.
5. The reward function scores the action.
6. The state manager advances the day/phase and updates short-term memory.

### Episode Structure

- `90` days per episode
- `3` phases per day
- phases: `morning`, `active`, `closing`

### Main Forces in the Simulation

- student demand clusters
- competitor shops and competitor pressure
- seasonal trends
- random events such as inflation and supply shortage
- inventory thresholds and auto-restocking
- awareness and satisfaction carryover

---

## Reward Function

The current reward is designed to be more profit-aligned than a simple revenue score.

At a high level, reward is shaped by:

- normalized gross profit
- satisfaction improvement
- inventory balance penalties
- overstock penalties
- controllable stockout penalties
- overpricing penalties

The implementation lives in:

- [`engine.py`](./campus_market_env/server/engine.py)
- [`config.py`](./campus_market_env/config.py)

This makes the environment suitable for experimenting with:

- reward shaping
- business-objective alignment
- long-horizon policy behavior
- tradeoffs between operational and customer-facing metrics

---

## API Reference

### Endpoints

- `GET /api/health`
- `POST /api/reset`
- `POST /api/step`
- `GET /api/state`

### Main Response Fields

Each observation includes:

- `day`
- `phase`
- `shop_traffic`
- `conversion_rate`
- `revenue`
- `customer_satisfaction`
- `inventory_level`
- `monthly_budget`
- `awareness`
- `market_sentiment`
- `competitor_pressure`
- `reward`
- `done`
- `info`

---

## Project Structure

```text
.
|-- campus_market_env/
|   |-- client.py
|   |-- config.py
|   |-- enums.py
|   |-- models.py
|   |-- openenv.yaml
|   `-- server/
|       |-- app.py
|       |-- competitor_model.py
|       |-- engine.py
|       |-- environment.py
|       |-- state_manager.py
|       |-- student_model.py
|       `-- trend_model.py
|-- docs/
|-- static/
|-- Dockerfile
|-- inference.py
|-- main.py
|-- requirements.txt
`-- test_env.py
```

### Important Files

- `campus_market_env/server/engine.py`: simulation math and reward computation
- `campus_market_env/server/environment.py`: reset/step environment wrapper
- `campus_market_env/server/state_manager.py`: phase/day transitions and memory
- `campus_market_env/server/student_model.py`: demand generation
- `campus_market_env/server/competitor_model.py`: competitor generation and pressure
- `campus_market_env/server/trend_model.py`: seasonality and trend logic
- `campus_market_env/models.py`: actions, observations, and state models
- `campus_market_env/config.py`: tunable constants

---

## Design Notes

Some deliberate design choices in this environment:

- deterministic seeded behavior for reproducible experiments
- modular simulation logic separated from the API layer
- explicit budget tracking and action clipping
- interpretable debug information returned in `info`
- reward shaping aimed at business realism rather than pure revenue maximization

---

## Suggested Workflows

### For RL Research

- train a policy against the HTTP or Python environment
- compare reward variants
- evaluate across multiple seeds
- inspect `info` for profit and action-execution debugging

### For LLM Agent Testing

- use [`inference.py`](./inference.py) as a baseline loop
- prompt a model with the current observation
- let it choose business actions
- compare cumulative reward across seeds

---

## Documentation

Additional docs:

- [`docs/GETTING_STARTED.md`](./docs/GETTING_STARTED.md)
- [`docs/QUICK_REFERENCE.md`](./docs/QUICK_REFERENCE.md)
- [`docs/IMPLEMENTATION_STATUS.md`](./docs/IMPLEMENTATION_STATUS.md)

---

## License

This repository currently does not expose a separate `LICENSE` file in the root. Add one if you want the project to be clearly reusable by others.
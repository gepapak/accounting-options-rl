# Prototype5 Renewable Electricity Market Simulator Engine

Part of the codebase for the paper *"Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
`engine/` contains the core Prototype5 simulation engine for renewable energy trading funds operating in European electricity markets (specifically the Nordic synchronous area, DK1/DK2). The simulator couples day-ahead wholesale forecasting with real-time continuous imbalance settlement.

## Key Components
- `financial_engine.py`: Handles cash accounting, mark-to-market valuations, capital distributions, and cash floor rules.
- `environment.py`: Gym-compatible multi-agent environment tracking market states, order execution, and settlement queues.
- `trading_engine.py`: Execution logic for buy/sell orders, balancing adjustments, and physical battery/renewable constraints.
- `marl.py` & `metacontroller.py`: Multi-Agent Proximal Policy Optimization (MAPPO) architecture and decentralized execution wrappers.
- `forecast_engine.py` & `forecast_price_experts.py`: Price and renewable generation prediction priors.
- `runtime_contract.py` & `protocol_preflight.py`: Pre-execution validation ensuring configuration parameters match protocol specifications.

## Usage
The engine is driven via the audit scripts in `audit/` and verified end-to-end by `tools/verify_release.py`.
```bash
python tools/verify_release.py
```
This recomputes every headline number and protocol configuration of the engine audit from the shipped run records.

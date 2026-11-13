# Market Data and Panel Generation

Part of the codebase for the paper *"Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
This directory contains real-world electricity market data from the Nordic synchronous system, covering wholesale day-ahead spot auctions and 15-minute imbalance settlement:
- `energinet/`: Raw and preprocessed price and volume series from Energinet Open Data (Denmark DK1 and DK2 price areas), licensed under CC BY 4.0.
- `esett/`: Nordic imbalance settlement time series from eSett Open Data.
- `engine_liquidity/`: Order book and market participation liquidity profiles.
- `derived/`: Pre-computed statistical moments, spread distributions, and train/test evaluation splits.
- `fetch_esett.py`: Automated fetcher that downloads eSett files across twelve Nordic zones directly from the source API and validates their SHA-256 digests.

## Data Provenance & Attribution
- Energinet data is published by Energinet under Creative Commons Attribution 4.0 International (CC BY 4.0).
- eSett data is sourced from the eSett Open Data portal. Raw eSett files are not redistributed; `fetch_esett.py` fetches the exact query ranges used in the study.
- Complete licensing terms and attribution details are provided in `THIRD_PARTY_LICENSES.md`.

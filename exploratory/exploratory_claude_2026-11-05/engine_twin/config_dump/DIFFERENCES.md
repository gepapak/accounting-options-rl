# Runtime settings that differ from L0 (MAPPO seed 7, DK1), read from the engine at its first step

## L1_price_taker

horizon settlement: True; prices differ from L0: False

- `liquidity_participation_cap_fraction`: 0.25 -> 0.0

## L2_no_solvency

horizon settlement: True; prices differ from L0: False

- `enable_trading_sleeve_margin`: True -> False
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0

## L3_price_taker_no_solvency

horizon settlement: True; prices differ from L0: False

- `enable_trading_sleeve_margin`: True -> False
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0

## L4_load_proxy_liquidity

horizon settlement: True; prices differ from L0: False

- `impact_ref_notional`: 'volume' -> 'sleeve'
- `liquidity_volume_source`: 'impact_volume' -> 'load'

## L5_percent_payoff

horizon settlement: True; prices differ from L0: False

- `mtm_horizon_payoff_denominator_mode`: 'mwh_volume' -> 'reference_price'

## L6_interpolated_prices

horizon settlement: True; prices differ from L0: True

- (no setting differs)

## L7_thesis_like

horizon settlement: False; prices differ from L0: True

- `distribution_rate`: 0.0 -> 0.1
- `enable_trading_sleeve_margin`: True -> False
- `eval_distribution_rate`: 0.0 -> 0.1
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L7_minus_interp

horizon settlement: False; prices differ from L0: False

- `distribution_rate`: 0.0 -> 0.1
- `enable_trading_sleeve_margin`: True -> False
- `eval_distribution_rate`: 0.0 -> 0.1
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L7_minus_percent_mtm

horizon settlement: True; prices differ from L0: True

- `distribution_rate`: 0.0 -> 0.1
- `enable_trading_sleeve_margin`: True -> False
- `eval_distribution_rate`: 0.0 -> 0.1
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0

## L7_minus_price_taker

horizon settlement: False; prices differ from L0: True

- `distribution_rate`: 0.0 -> 0.1
- `enable_trading_sleeve_margin`: True -> False
- `eval_distribution_rate`: 0.0 -> 0.1
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L7_minus_no_solvency

horizon settlement: False; prices differ from L0: True

- `distribution_rate`: 0.0 -> 0.1
- `eval_distribution_rate`: 0.0 -> 0.1
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L7_minus_legacy_fee

horizon settlement: False; prices differ from L0: True

- `distribution_rate`: 0.0 -> 0.1
- `enable_trading_sleeve_margin`: True -> False
- `eval_distribution_rate`: 0.0 -> 0.1
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L7_minus_sweeper

horizon settlement: False; prices differ from L0: True

- `enable_trading_sleeve_margin`: True -> False
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L8_pctmtm_nosolv

horizon settlement: False; prices differ from L0: False

- `enable_trading_sleeve_margin`: True -> False
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

## L7_thesis_like_capon

horizon settlement: False; prices differ from L0: True

- `distribution_rate`: 0.0 -> 0.1
- `enable_trading_sleeve_margin`: True -> False
- `eval_distribution_rate`: 0.0 -> 0.1
- `liquidity_participation_cap_fraction`: 0.25 -> 0.0
- `market_fee_model`: 'nord_pool_intraday_2026' -> 'legacy_notional_fixed'
- `mtm_apply_price_return_cap`: False -> True
- `mtm_loss_exit_threshold_pct`: 0.15 -> 1000.0
- `mtm_return_model`: 'horizon_settlement' -> 'percent_capped'

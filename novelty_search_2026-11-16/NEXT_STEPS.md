# Decisions for the author (16 November 2026)

1. **DONE: Study XE was registered on the author's approval** (hashed 2026-11-16T19:46:10Z) and is running in
   `../study_xe_2026-11-16/`. Earlier wording, kept for the record: a ready-to-run package was in `../study_xe_draft/`. It contains the protocol with fixed fresh seeds, decisions
   XE1 to XE7 with thresholds from batches 1 to 6, the exact commands, a decision script dry-run on pilot data, and a hash script. It
   is not hashed yet. On your approval it is hashed and run, about 4 to 4.5 hours overnight. It turns the exploratory cross-engine and
   PPO-through-engine results into registered evidence the paper can lead with.
2. **Integrate into the manuscript?** Draft sentences are in `INTEGRATION_NOTES.md`: three to four sentences in "Where the conditions
   arise" and one supplementary section. That is a small page cost, but the paper is already over EAAI's limit.
3. **Report upstream to backtesting.py?** This is outward-facing, so it is the author's call. An issue would describe the zero floor
   with the one-query example from `reading_engines.py`. Related open threads: #939, #1441, #1429.
4. **An LLM-agent demonstration (optional, needs an API key and a budget)?** An LLM agent that writes and tunes strategies against
   backtesting.py's reported return. Prior art already shows LLMs exploit limited liability in prompts (Jarne Ornia et al. 2025), so
   the new part would only be the agent-plus-engine loop. In-context reward hacking through feedback loops is also known (Pan et al.,
   ICML 2024), so this would be one more instance. It is not recommended as a novelty source.

Not recommended: more pivots. Each direction checked today is listed in `README.md`, with its result.

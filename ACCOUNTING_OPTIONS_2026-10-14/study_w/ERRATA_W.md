# Study W errata (3 November 2026)

`PREREGISTRATION_W.md` is a hashed, registered file and stays byte-for-byte unchanged; this note corrects one statement of its background and
documents one ambiguity of wording found by the fourth AI-assisted review (Codex, 3 November 2026). No registered decision or verdict changes.

1. **"that strengths 1 and 25 are not"** (line 9). In Study V, strength 25 met both criteria in Part A (exploit share 0.00, 84 percent of the
   ledger-trained edge) and failed only in Part C (from zero equity it stopped trading: 21 percent of the edge). Strength 1 failed in both parts.
   The correct summary is: strength 5 did both in both starting-equity regimes, strength 25 did both with positive starting equity only, and
   strength 1 in neither.
2. **Two meanings of "removes the exploit".** Study W's registered W2 judges a strength by arm means only (mean exploit share at most 0.10 and
   mean booked edge at least 80 percent of the ledger-trained arm's). The common definition of Studies S to X (and Y) additionally requires a
   one-sided Mann-Whitney test against the reported-reward arm (`p < 0.05`). Against Study T's reported-reward runs, strength 1 in DK2 has
   mean exploit share 0.078 but `p = 0.076`, so it meets W's mean-based criterion and not the significance-based one. The W verdicts are unchanged;
   the manuscript states which criterion each sentence uses (`exploratory_claude_2026-11-03/w_common_criterion.py` computes every cell).

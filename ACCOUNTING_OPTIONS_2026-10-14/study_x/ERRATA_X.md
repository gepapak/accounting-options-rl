# Study X errata (3 November 2026)

`PREREGISTRATION_X.md` is a hashed, registered file and stays byte-for-byte unchanged; this note corrects one statement of its background
paragraph ("Theory (exploratory)"). It enters no hypothesis, decision rule or verdict, and no verdict of Study X changes.

**"(a small overpayment, no asymmetry between branches, and zero in a fair market)"** (line 18). For an account with equity `e` below the fee
`c` that writes the fee first with its own floor, TIR returns `g - e` on the favourable branch (booked `g - c`) and `-g - 2c + e` on the
unfavourable one (booked `-g - c`): the errors are `+(c - e)` and `-(c - e)`, equal in absolute value with opposite signs, not the same on both
branches. The expected overpayment `(2p - 1)(c - e)` stated in the same sentence is correct and is zero in a fair market. Found by the fourth
AI-assisted review (Codex, 3 November 2026).

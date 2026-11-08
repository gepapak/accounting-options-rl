# Study T errata (3 November 2026)

`PREREGISTRATION_T.md` is a hashed, registered file and stays byte-for-byte unchanged; this note corrects two statements of its background
paragraph ("Theory (exploratory)"). Neither statement enters a hypothesis, a decision rule or a verdict, and no verdict of Study T changes.

1. **"SPR is the sharp lower end of that interval"** (line 20) is false. SPR's magnitude `|r - r'|/2` is a valid lower bound on the gross
   magnitude `g` but in general not the sharp one: with `g = 10`, `c = 1` and `tau = 0` the pair is `(9, 0)`, which identifies `g = 10`, while SPR
   uses 4.5. The sharp lower end of the identified set is TIR's magnitude (the identified value where `r != r'`, zero where `r = r'`); see the
   manuscript's propositions "Twin identification" and "The twin-identified reward". Found by the third AI-assisted review (1 November 2026).
2. **"Under every rule of the accounting ladder the reported step reward is a left-censored observation of the booked P&L"** (lines 17-18)
   overstates the scope: the reported increment of an owner's claim while insolvent (Study N's zombie window under a settlement lag) censors the
   booked P&L minus the carried deficit, and simulators that floor several cash writes in sequence have no single threshold. The manuscript's
   Assumption C and its two stated exceptions give the correct scope. Found by the fourth AI-assisted review (3 November 2026).

# Nordic settlement rules used in the extension (checked 14 Oct 2026)

Quotes are kept short; the full texts are in the linked documents. `esett_app2.txt` and `forskrift_c1.txt` in this folder are local text extractions of the two PDFs; they are working copies only and are not for redistribution.

1. **The eSett collateral formula.** Source: eSett, *Imbalance Settlement Agreement, Appendix 2 Collaterals*, 21 Aug 2023, https://www.esett.com/app/uploads/2023/08/Imbalance-Settlement-Agreement-Appendix-2-Collaterals_21.8.2023.pdf
   - The collateral requirement is calculated each Monday. eSett may recalculate it during the week if counterparty risk "increases significantly" (3.1).
   - The Standard Formula has the form C = 3(S1 + S2) + m(V1 + V2)P:
     - S1: average weekly fees over the last three invoiced weeks;
     - S2: average of the absolute weekly sums of invoiced imbalances over the last three invoiced weeks;
     - V1: consumption volume over the last seven settled days;
     - V2: bilateral and PX sales volumes from day −8 to day −2;
     - m: 3/7 up to 80,000 MWh, 1/7 up to 400,000 MWh, 0 above;
     - P: the last seven days' average imbalance price, with negative prices set to zero.
   - Minimum: 40,000 EUR per country in Finland, Norway and Sweden.
   - "The Collateral Requirement in Denmark is 0 EUR" (3.3).
   - eSett may apply situation-specific or BRP-specific formulas (3.4–3.5). Collateral is due the same day the requirement is published (4.4).
2. **Settlement timing and who bears the Danish risk.** Source: eSett, *Information to prospective settlement banks*, 16 Sep 2025, https://www.esett.com/app/uploads/2025/09/Information-to-prospective-settlement-banks_16.9.2025.pdf
   - "The imbalance settlement results are invoiced and settled ~3 weeks after the delivery day."
   - Denmark is excepted from eSett collateral, being the area "where Energinet (TSO) bares the risk".
   - Minimum collateral demand is EUR 40,000.
3. **Energinet's security rules.** Source: Energinet, *Forskrift C1: Vilkår for balanceansvar*, version of 28 Jan 2021 (now marked historical), https://energinet.dk/media/1h4nrmdn/forskrift-c1-januar-2021.pdf
   - The BRP must be covered by an Energinet-approved credit insurance or post an on-demand guarantee (§3 stk. 2).
   - Credit limits:
     - consumption BRPs: 5 % of the previous month's consumption × the year's forward price fixed by Nord Pool in November of the preceding year (stk. 4);
     - trading BRPs: fixed at DKK 2,000,000 (stk. 6);
     - minimum: DKK 2,000,000 nationally (stk. 3).
   - The credit limit is defined as the participant's expected maximum outstanding amount at normal settlement frequency.
   - The balance agreement appended to C1 says Energinet normally uses credit insurance, so the participant usually does not post security.
   - Energinet's current English BRP page reportedly asks for DKK 1,000,000 via credit assessment or bank guarantee. This was seen only in a search snippet; the page text could not be retrieved.
4. **Finland's 2023 collateral decision.** Source: Fingrid news, 20 Dec 2023, https://www.fingrid.fi/en/news/news/2023/energy-authority-publishes-decision-on-the-collateral-model-for-balance-responsible-parties/
   - The Energy Authority decided on a collateral model that departs from the shared Swedish–Norwegian model.
   - Under it, negative prices are replaced by zero, the multiplier is the same for all BRPs, and the period covered by the collateral is shortened.
   - It must take effect by 30 Nov 2024.

## Modelling caveats

- **Deliberate imbalance.** Balance agreements oblige BRPs to plan to be in balance, and TSOs monitor intentional imbalances. Our "open exposure" is a stress-test quantity: the imbalance of a BRP that stops hedging its consumption. It is not a recommended strategy.
- **Franchise value.** Distressed firms with franchise value may cut risk instead of gambling (Kirti, JBF 2024).
- **Collateral sizing.** Collateral rules are sized for credit exposure on the full energy value. Our comparison asks a narrower question: does posted collateral make a no-edge limited-liability gamble on the spread unprofitable?
- **Cost of risk.** The "cost" c is set to 0.10 EUR/MWh, the paper's fee; results at 0.5 EUR/MWh are also reported. Actual marginal costs of carrying an imbalance were not verified.
- **Danish prices.** The eSett formula applies in FI/SE/NO. Applying it to Danish prices is a stylised comparison; Finnish eSett data would make it direct.

5. **Post-reform volatility, as eSett itself reports.** Source: eSett, *Nordic Electricity Market – Q1 2026*, https://www.esett.com/news/nordic-electricity-market-q1-2026/
   - Imbalance prices remained volatile, "with price spikes occurring at regular intervals across all Nordic countries".
   - "The volatility has persisted since the introduction of the mFRR EAM in March 2025."
   - The Q3 2025 report (https://www.esett.com/news/nordic-electricity-market-q3-2025/) describes significant volatility after the launch.
   - Mitigations reported by TSOs and in industry commentary (seen in search summaries only; not checked against primary sources):
     - Energinet: a 25 MW tolerance band from 2 June 2025;
     - Fingrid: a floating 50 MW deadband on the mFRR request.
   - These match our data (`tails12.csv`, 2023-01..2025-03 vs post-reform). Spread volatility rose in the body and the tails while price levels did not. Median absolute spreads rose two- to sevenfold in the Danish, Swedish and Finnish zones (DK1 5.3 → 38.4 EUR/MWh) and stayed below 9 EUR/MWh in Norway. Quarter-hours with |spread| > 1,000 EUR/MWh became much more frequent: DK1 from 2.4 to 40.7 per 10,000, NO4 from 0.3 to 11.6, NO3 from 0 to 6.4, SE1 from 0.3 to 4.9.

# Glossary

One term per line, shown on the Study tab:

**teacher** — the rollout search that checks the advisor's pick at buys and action plays (from bv-v1)
**c2** — RL-v3's best net (iteration 2654): proposes candidates and plays both sides in every rollout
**rollout** — one copy of the game played to the end (or to a leaf) by c2 after one candidate move
**draw** — one deal of the hidden cards; every candidate is played on the same draw (paired)
**public sampler** — deals the opponent's unseen cards from what a player can know (obs 1)
**leaf (D)** — stop a rollout D turns after the decision and score it with c2's win estimate
**successive halving** — spend the rollouts in rounds, dropping the worse half of the candidates each round
**margin / z** — the override gate: gain above the margin and above z standard errors of the paired difference
**arm** — one match configuration in queue.txt, played on the same 300 kingdoms × 2 seats as every other arm
**paired score** — mean points over both seats of a kingdom; intervals are bootstrapped over kingdoms

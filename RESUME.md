# Paused 2026-11-25 19:15 (user asked to stop; resume here)

Everything was stopped cleanly; finished games and bank settings are saved. Nothing runs (no ts-* units, the
publish loop is stopped, the site says so).

## Where it stands
- Advisor (autopilot, committed 5da4a78): teacher = this repo's ts.so, 3 processes pooled per option, successive
  halving, 256 draws/option/process within 3 s, gate margin 0.01 / z 1, card choices (Workshop / Artisan gain,
  Remodel / Mine trash + gain, Moneylender, Throne Room); judge = c2. Replay of session2 with these defaults: 26
  answers fine, then a teacher-thread crash on recalibration -- fixed in 5da4a78, the full replay was NOT rerun yet.
- Bank (results/bank_eval.json): c2 0.88 regret/decision; today's old teacher 0.57; 320 draws + halving + new gate 0.31.
- Matches vs c2 (stream ts1): A (old teacher) ~62%, stopped at 215 kingdoms; G (A + card choices) +5.2 points
  [+0.2, +10.1] over A on 111 kingdoms; I (the advisor's new teacher) had just started (3 games).
- Rollouts 1.6x faster, bit-identical (engine b49a11e, rl-v3 56964d0, c/fastnet.h).

## Resume, in this order
1. Rerun the advisor replay: `cd ../../autopilot && .venv/bin/python autopilot/advisor_bench.py recordings/session2`
   (expect no "Exception in thread teacher"; DOMINION_TEACHER_DUMP=<file> logs every answer).
2. `systemd-run --user --unit=ts-yield --working-directory=$PWD experiments/yield.sh` (pauses arms while you play).
3. Arms: `experiments/arm.sh I_full_k320_sh_m01z1_sub 6` and `experiments/arm.sh G_full_k128_sub 2` (both resume).
4. Bank: `.venv/bin/python experiments/bank_eval.py --workers 2 --configs full640` (the only setting not finished).
5. Site: `setsid nohup ./publish_loop.sh > /dev/null 2>&1 < /dev/null &`.

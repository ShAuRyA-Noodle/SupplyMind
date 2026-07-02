"""Live Counterfactual Digital Twin.

Given a live Hormuz (or arbitrary crisis) signal, simulate 100 Monte-Carlo
rollouts of: (a) trained MaskablePPO, (b) no-action baseline, (c) greedy
baseline. Return the loss distribution (not just a point estimate) and a
headline "$ saved vs no-action" number conditioned on the live signal.

Makes the v4 scripted "$324M -> $65M = 80% savings" into a live, run-anytime
computation tied to today's NewsAPI + FRED Brent reading.
"""

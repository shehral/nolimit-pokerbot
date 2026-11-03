# Short deck (36 cards)
Deck is 6 through ace (36 cards), everything else as plain hold'em (2 hole cards, 5-card board, 8 seats).
Ranking, high to low: straight flush, four of a kind, FLUSH, FULL HOUSE, straight, three of a kind, two pair, pair,
high card. A-6-7-8-9 is a straight (the lowest; 6-7-8-9-T beats it). Variant option `trips_beat_straight=True` puts
three of a kind above a straight. Hole-card classes: 81 (not 169).
Evaluator: `sim.variants.evals.evaluate_short`.

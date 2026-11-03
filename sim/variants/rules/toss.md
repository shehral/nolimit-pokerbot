# Toss (rehearsal rules, MIT 2026 "Toss or Hold'em" generalised to 8 seats)
8-max no-limit sit-and-go, blinds and payouts as the plain kit.
1. Each player gets THREE hole cards. Betting preflop as usual.
2. The flop is only TWO cards. Before flop betting every player still in the hand (all-in players too) tosses one hole
   card face up to the board, one after another, starting left of the button. Later players see earlier tosses.
3. The tossed card is a normal board card. After all tosses the board has 2 + (number of players) cards; flop betting,
   then turn and river (one card each). Showdown: best five of your two kept cards plus the whole board.
Interface: `state["request"] == "discard"`, `legal_actions == ["discard"]`, `discard_options` = indexes into `hole`.
Answer `{"action": "discard", "index": k}`. An illegal answer, exception or over-1 ms answer folds you.

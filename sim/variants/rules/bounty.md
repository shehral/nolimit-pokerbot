# Bounty (MIT 2025 "Bounty Hold'em", 8-seat adaptation)
Plain hold'em betting. At the start of every blind level (every 25 hands) each player is secretly given a bounty rank
(uniform, repeats allowed; `state["bounty_rank"]` is your own). When you win a pot and your bounty rank is among your
hole cards or on the board (even if everybody folded), your winnings become 1.5 x normal winnings + 10, where normal
winnings = the chips the others put into the pot you won. The extra comes from those players, in proportion to what they
put in, and never exceeds the chips they have left. Chopped pot: exactly one hit pays normal/4 + 10 (two half-pot
boards). Fractions round up unless you are the dealer. Opponents' ranks are never shown; `bounty_bonus` events show the
chips moved.

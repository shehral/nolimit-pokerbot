# Antes
`antes`: from blind level 2 every player posts an ante of 0.125 big blinds (rounded to whole chips, minimum 1) before the
blinds; antes are dead money (in the pot, not in the call amount). `bbante`: from level 2 only the big blind seat posts one
extra big blind as the ante. Level 1 has no ante. `state["ante"]` is the per-hand ante amount; `history_this_hand` lists
the ("ante") postings. Everything else is plain hold'em.

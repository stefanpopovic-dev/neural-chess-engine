# Match: nn-cache vs material

100 games, 0.5s per move, move cap 200 plies (scored as a draw), 4 parallel game(s), 50 openings x 2 colours.

| engine A |   W |   D |   L | score | Elo diff (A - B) | 95% CI       | LOS    |
| -------- | --: | --: | --: | ----- | ---------------- | ------------ | ------ |
| nn-cache |  73 |  21 |   6 | 83.5% | +282 +/- 75      | [+218, +367] | 100.0% |

As White: {'wins': 33, 'draws': 15, 'losses': 2}; as Black: {'wins': 40, 'draws': 6, 'losses': 4}.
Terminations: {'checkmate': 79, 'move_cap': 18, 'threefold_repetition': 3}; average game length 115 plies.

| engine   | avg depth | nodes/sec | evals/move | net calls/move | cache hit rate |
| -------- | --------: | --------: | ---------: | -------------: | -------------- |
| nn-cache |      2.95 |    23,865 |      6,628 |          4,206 | 35.6%          |
| material |      4.00 |    60,417 |     17,191 |              0 | -              |

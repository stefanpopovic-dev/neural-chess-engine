Search profile: 5s on `ruy_lopez_closed`, wall-clock timers around the evaluator.

| evaluator | nodes/sec | depth |   evals | net calls | time in evaluate() | time in static eval / network | us per eval | nodes/sec if eval free |
| --------- | --------: | ----: | ------: | --------: | ------------------ | ----------------------------- | ----------: | ---------------------: |
| material  |    74,259 |     5 | 236,541 |         0 | 26.7%              | 4.9%                          |        5.70 |                101,363 |
| nn        |    18,709 |     4 |  63,352 |    63,352 | 73.1%              | 64.7%                         |       57.70 |                 69,644 |
| nn-cache  |    23,103 |     4 |  76,152 |    53,535 | 72.3%              | 54.4%                         |       47.50 |                 83,310 |

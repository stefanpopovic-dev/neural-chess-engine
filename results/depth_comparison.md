Depth benchmark at 1s/move, 12 positions x 3 moves (search: quiescence=True).

| config   | depth (all) | depth open/mid/end | nodes/sec | evals/move | net calls/move | cache hit rate |
| -------- | ----------: | ------------------ | --------: | ---------: | -------------: | -------------- |
| material |        5.22 | 4.4/4.0/7.2        |    80,159 |     44,598 |              0 | -              |
| nn       |        3.92 | 3.2/3.2/5.3        |    24,928 |     13,686 |         13,677 | -              |
| nn-cache |        4.17 | 3.4/3.2/5.8        |    39,120 |     20,425 |          8,054 | 60.5%          |

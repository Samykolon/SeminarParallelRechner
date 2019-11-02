Floyd-Marshall

Seriell

7: 48 ms
8: 389 ms
9: 3252 ms
10: 24592 ms

Blockgröße 16x16

1 GPU - 1 Rank 

7: 54 ms
8: 93 ms
9: 448 ms
10: 2002 ms
11: 11494 ms
12: 89402 ms

1 GPU - 4 Ranks

7: 125 ms
8: 135 ms
9: 1139 ms
10: 1362 ms
11: 7988 ms
12: 61916 ms

3 GPUs - 3 Ranks

7: 80 ms
8: 183 ms
9: 436 ms
10: 1539 ms
11: 4106 ms
12: 22491 ms

4 GPUs - 4 Ranks

7: 114 ms
8: 241 ms
9: 535 ms
10: 1488 ms
11: 4712 ms
12: 17067 ms

Blockgröße 256x256

1 GPU - 1 Rank - Blockgröße 16x16

7: 10 ms
8: 14 ms
9: 53 ms
10: 113 ms
11: 280 ms
12: 839 ms

1 GPU - 4 Ranks

7: 91 ms
8: 135 ms
9: 215 ms
10: 388 ms
11: 722 ms
12: 1620 ms

3 GPUs - 3 Ranks

7: 73 ms
8: 150 ms
9: 464 ms
10: 1201 ms
11: 3555 ms
12: 11794 ms

4 GPUs - 4 Ranks

7: 105 ms
8: 218 ms
9: 523 ms
10: 1343 ms
11: 4366 ms
12: 14390 ms

Vector-Add - Unified Shared Memory

1 GPU - 1 Rank

5 000 000: 288 ms (26.3 ms Kernel)
10 000 000: 516 ms (44.1 ms Kernel)
15 000 000: 743 ms (51.3 ms Kernel)
20 000 000: 977.4 ms (74.5 ms Kernel)

1 GPU - 4 Ranks

5 000 000: 1158.2 ms (7.9 ms Kernel)
10 000 000: 1850.6 ms (22.5 ms Kernel)
15 000 000: 2746.2 ms (48.7 ms Kernel)
20 000 000: 4670.4 ms (52.5 ms Kernel)

3 GPUs - 3 Ranks

5 000 000: 7405.28 ms (24.8 ms Kernel)
10 000 000: 14691.6 ms (34.3 ms Kernel)
15 000 000: 22092.2 ms (67.6 ms Kernel)
20 000 000: 29351.8 ms (70.2 ms Kernel)

4 GPUs - 4 Ranks

5 000 000: 10935.5 ms (27.1 ms Kernel)
10 000 000: 21829.1 ms (32.4 ms Kernel)
15 000 000: 32592.9ms (45.2 ms Kernel)
20 000 000: 43554.5 ms (62.7 ms Kernel)




TSP-Algorithmus

Cuda-Only:

8 Städte
7 Threads, 720 Kombinationen, 61.3 ms

9 Städte
8 Threads, 40320 Kombinationen, 118.7 ms

10 Städte
72 Threads, 362880 Kombinationen, 359.5 ms

11 Städte
720 Threads, 3628800 Kombinationen, 444.6 ms

12 Städte
990 Threads * 8 Blöcke, 39916800 Kombinationen, 1628.1 ms

13 Städte
495 Threads * 24 Blöcke * 8 Kernel, 479001600 Kombinationen, 18340 ms

14 Städte
715 Threads * 27 Blöcke * 64 Kernel, 6227020800 Kombinationen, 276130.4 ms

Cuda-Only - Powermodus

8 Städte
7 Threads, 720 Kombinationen, 11.3 ms

9 Städte
8 Threads, 40320 Kombinationen, 40.1 ms

10 Städte
72 Threads, 362880 Kombinationen, 51.9 ms

11 Städte
720 Threads, 3628800 Kombinationen, 131.2 ms

12 Städte
990 Threads * 8 Blöcke, 39916800 Kombinationen, 1442 ms

13 Städte
495 Threads * 24 Blöcke * 8 Kernel, 479001600 Kombinationen, 17740.5 ms

14 Städte
715 Threads * 27 Blöcke * 64 Kernel, 6227020800 Kombinationen, 256130.4 ms


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




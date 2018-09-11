# Benchmark Report for */home/bates/.julia/dev/MixedModels/src/MixedModels.jl*

## Job Properties
* Time of benchmark: 11 Sep 2018 - 15:12
* Package commit: dirty
* Julia commit: 5d4eac
* Julia command flags: None
* Environment variables: None

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                                                                                                                                                  | time            | GC time    | memory           | allocations |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------:|-----------------:|------------:|
| `["crossed", "Assay:1+A+B*C+(1|G)+(1|H)"]`                                                                                                                                                          |   2.299 ms (5%) |            |  353.95 KiB (1%) |        4734 |
| `["crossed", "Demand:1+U+V+W+X+(1|G)+(1|H)"]`                                                                                                                                                       |   1.973 ms (5%) |            |  263.61 KiB (1%) |        4859 |
| `["crossed", "InstEval:1+A*I+(1|G)+(1|H)"]`                                                                                                                                                         |    1.283 s (5%) | 115.194 ms |  234.32 MiB (1%) |       26746 |
| `["crossed", "InstEval:1+A+(1|G)+(1|H)+(1|I)"]`                                                                                                                                                     |    2.020 s (5%) |  15.281 ms |  186.83 MiB (1%) |       30568 |
| `["crossed", "Penicillin:1+(1|G)+(1|H)"]`                                                                                                                                                           |   1.816 ms (5%) |            |  215.63 KiB (1%) |        3688 |
| `["crossed", "ScotsSec:1+A+U+V+(1|G)+(1|H)"]`                                                                                                                                                       |   3.873 ms (5%) |            |    1.32 MiB (1%) |        5431 |
| `["crossed", "dialectNL:1+A+T+U+V+W+X+(1|G)+(1|H)+(1|I)"]`                                                                                                                                          | 418.403 ms (5%) |   5.788 ms |   94.79 MiB (1%) |       14801 |
| `["crossed", "egsingle:1+A+U+V+(1|G)+(1|H)"]`                                                                                                                                                       |  29.704 ms (5%) |   4.374 ms |   48.09 MiB (1%) |       12963 |
| `["crossed", "ml1m:1+(1|G)+(1|H)"]`                                                                                                                                                                 |   36.889 s (5%) | 196.658 ms |  472.85 MiB (1%) |     4976959 |
| `["crossed", "paulsim:1+S+T+U+(1|H)+(1|G)"]`                                                                                                                                                        |  12.989 ms (5%) |            |    4.28 MiB (1%) |        5842 |
| `["crossedvector", "bs10:1+U+V+W+((1+U+V+W)|G)+((1+U+V+W)|H)"]`                                                                                                                                     | 157.970 ms (5%) |   2.611 ms |   24.47 MiB (1%) |      773227 |
| `["crossedvector", "d3:1+U+((1+U)|G)+((1+U)|H)+((1+U)|I)"]`                                                                                                                                         |   52.687 s (5%) |    1.639 s |    7.62 GiB (1%) |   303945264 |
| `["crossedvector", "d3:1+U+(1|G)+(1|H)+(1|I)"]`                                                                                                                                                     | 304.071 ms (5%) | 118.329 ms |  371.45 MiB (1%) |       34328 |
| `["crossedvector", "gb12:1+S+T+U+V+W+X+Z+((1+S+U+W)|G)+((1+S+T+V)|H)"]`                                                                                                                             | 122.401 ms (5%) |            |   14.48 MiB (1%) |      491328 |
| `["crossedvector", "kb07:1+S+T+U+V+W+X+Z+((1+S+T+U+V+W+X+Z)|G)+((1+S+T+U+V+W+X+Z)|H)"]`                                                                                                             |    3.979 s (5%) |  10.894 ms |  118.38 MiB (1%) |     4113879 |
| `["crossedvector", "kb07:1+S+T+U+V+W+X+Z+(1|G)+((0+S)|G)+((0+T)|G)+((0+U)|G)+((0+V)|G)+((0+W)|G)+((0+X)|G)+((0+Z)|G)+(1|H)+((0+S)|H)+((0+T)|H)+((0+U)|H)+((0+V)|H)+((0+W)|H)+((0+X)|H)+((0+Z)|H)"]` | 496.175 ms (5%) |   6.625 ms |   69.97 MiB (1%) |     3210225 |
| `["nested", "Animal:1+(1|G)+(1|H)"]`                                                                                                                                                                | 961.698 μs (5%) |            |  131.61 KiB (1%) |        2285 |
| `["nested", "Chem97:1+(1|G)+(1|H)"]`                                                                                                                                                                |  60.214 ms (5%) |   8.228 ms |   93.67 MiB (1%) |       16366 |
| `["nested", "Chem97:1+U+(1|G)+(1|H)"]`                                                                                                                                                              |  61.120 ms (5%) |   8.229 ms |   94.44 MiB (1%) |       16639 |
| `["nested", "Genetics:1+A+(1|G)+(1|H)"]`                                                                                                                                                            |    6.786 s (5%) | 134.644 ms |  483.45 MiB (1%) |     9576601 |
| `["nested", "Pastes:1+(1|G)+(1|H)"]`                                                                                                                                                                |   1.623 ms (5%) |            |  218.94 KiB (1%) |        3534 |
| `["nested", "Semi2:1+A+(1|G)+(1|H)"]`                                                                                                                                                               |   1.652 ms (5%) |            |  247.36 KiB (1%) |        3844 |
| `["simplescalar", "Alfalfa:1+A*B+(1|G)"]`                                                                                                                                                           | 990.919 μs (5%) |            |  181.16 KiB (1%) |        2623 |
| `["simplescalar", "Alfalfa:1+A+B+(1|G)"]`                                                                                                                                                           | 847.247 μs (5%) |            |  146.17 KiB (1%) |        2170 |
| `["simplescalar", "AvgDailyGain:1+A*U+(1|G)"]`                                                                                                                                                      |   1.014 ms (5%) |            |  158.44 KiB (1%) |        2670 |
| `["simplescalar", "AvgDailyGain:1+A+U+(1|G)"]`                                                                                                                                                      | 901.207 μs (5%) |            |  138.23 KiB (1%) |        2268 |
| `["simplescalar", "BIB:1+A*U+(1|G)"]`                                                                                                                                                               |   1.170 ms (5%) |            |  169.19 KiB (1%) |        3007 |
| `["simplescalar", "BIB:1+A+U+(1|G)"]`                                                                                                                                                               | 912.081 μs (5%) |            |  138.14 KiB (1%) |        2299 |
| `["simplescalar", "Bond:1+A+(1|G)"]`                                                                                                                                                                | 764.849 μs (5%) |            |  117.00 KiB (1%) |        1821 |
| `["simplescalar", "Cultivation:1+A*B+(1|G)"]`                                                                                                                                                       | 903.114 μs (5%) |            |  149.36 KiB (1%) |        2511 |
| `["simplescalar", "Cultivation:1+A+(1|G)"]`                                                                                                                                                         | 850.280 μs (5%) |            |  125.20 KiB (1%) |        2047 |
| `["simplescalar", "Cultivation:1+A+B+(1|G)"]`                                                                                                                                                       | 888.806 μs (5%) |            |  140.30 KiB (1%) |        2348 |
| `["simplescalar", "Dyestuff2:1+(1|G)"]`                                                                                                                                                             | 641.758 μs (5%) |            |   80.81 KiB (1%) |        1427 |
| `["simplescalar", "Dyestuff:1+(1|G)"]`                                                                                                                                                              | 714.685 μs (5%) |            |   87.41 KiB (1%) |        1599 |
| `["simplescalar", "Exam:1+A*U+B+(1|G)"]`                                                                                                                                                            |   1.923 ms (5%) |            |    1.14 MiB (1%) |        3464 |
| `["simplescalar", "Exam:1+A+B+U+(1|G)"]`                                                                                                                                                            |   1.810 ms (5%) |            | 1022.16 KiB (1%) |        3124 |
| `["simplescalar", "Gasoline:1+U+(1|G)"]`                                                                                                                                                            | 862.989 μs (5%) |            |  123.28 KiB (1%) |        2028 |
| `["simplescalar", "Hsb82:1+A+B+C+U+(1|G)"]`                                                                                                                                                         |   2.770 ms (5%) |            |    2.09 MiB (1%) |        3768 |
| `["simplescalar", "IncBlk:1+A+U+V+W+Z+(1|G)"]`                                                                                                                                                      |   1.047 ms (5%) |            |  185.14 KiB (1%) |        3357 |
| `["simplescalar", "Mississippi:1+A+(1|G)"]`                                                                                                                                                         | 777.914 μs (5%) |            |  119.69 KiB (1%) |        1851 |
| `["simplescalar", "PBIB:1+A+(1|G)"]`                                                                                                                                                                |   1.145 ms (5%) |            |  188.47 KiB (1%) |        2379 |
| `["simplescalar", "Rail:1+(1|G)"]`                                                                                                                                                                  | 851.673 μs (5%) |            |   99.77 KiB (1%) |        1939 |
| `["simplescalar", "Semiconductor:1+A*B+(1|G)"]`                                                                                                                                                     |   1.089 ms (5%) |            |  193.50 KiB (1%) |        2710 |
| `["simplescalar", "TeachingII:1+A+T+U+V+W+X+Z+(1|G)"]`                                                                                                                                              |   1.271 ms (5%) |            |  255.69 KiB (1%) |        4525 |
| `["simplescalar", "cake:1+A*B+(1|G)"]`                                                                                                                                                              |   1.396 ms (5%) |            |  387.00 KiB (1%) |        2820 |
| `["simplescalar", "ergoStool:1+A+(1|G)"]`                                                                                                                                                           | 822.350 μs (5%) |            |  125.91 KiB (1%) |        1942 |
| `["singlevector", "Early:1+U+U&A+((1+U)|G)"]`                                                                                                                                                       |  20.686 ms (5%) |            |    3.31 MiB (1%) |       75262 |
| `["singlevector", "HR:1+A*U+V+((1+U)|G)"]`                                                                                                                                                          |   4.677 ms (5%) |            |  829.61 KiB (1%) |       25177 |
| `["singlevector", "Oxboys:1+U+((1+U)|G)"]`                                                                                                                                                          |  11.519 ms (5%) |            |    1.65 MiB (1%) |       42515 |
| `["singlevector", "SIMS:1+U+((1+U)|G)"]`                                                                                                                                                            |  64.582 ms (5%) |            |   12.61 MiB (1%) |      385749 |
| `["singlevector", "WWheat:1+U+((1+U)|G)"]`                                                                                                                                                          |   5.501 ms (5%) |            |  662.48 KiB (1%) |       16260 |
| `["singlevector", "Weights:1+A*U+((1+U)|G)"]`                                                                                                                                                       |  17.842 ms (5%) |            |    2.98 MiB (1%) |       85818 |
| `["singlevector", "sleepstudy:1+U+((1+U)|G)"]`                                                                                                                                                      |   4.127 ms (5%) |            |  693.59 KiB (1%) |       20434 |
| `["singlevector", "sleepstudy:1+U+(1|G)+((0+U)|G)"]`                                                                                                                                                |   2.853 ms (5%) |            |  550.09 KiB (1%) |       17385 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["crossed"]`
- `["crossedvector"]`
- `["nested"]`
- `["simplescalar"]`
- `["singlevector"]`

## Julia versioninfo
```
Julia Version 1.0.0
Commit 5d4eaca0c9 (2018-08-08 20:58 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 18.04.1 LTS
  uname: Linux 4.15.0-34-generic #37-Ubuntu SMP Mon Aug 27 15:21:48 UTC 2018 x86_64 x86_64
  CPU: Intel(R) Core(TM) i5-3570 CPU @ 3.40GHz: 
              speed         user         nice          sys         idle          irq
       #1  1650 MHz     339398 s    1507366 s      21535 s     116159 s          0 s
       #2  1646 MHz     371963 s    1478344 s      24252 s     110723 s          0 s
       #3  1668 MHz     347431 s    1510470 s      20648 s     104137 s          0 s
       #4  2100 MHz     345199 s    1512685 s      20893 s     105679 s          0 s
       
  Memory: 15.554649353027344 GB (1843.765625 MB free)
  Uptime: 19868.0 sec
  Load Avg:  1.58544921875  2.9697265625  4.3095703125
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.0 (ORCJIT, ivybridge)
```
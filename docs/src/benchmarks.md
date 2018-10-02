# Benchmark Report for */home/bates/.julia/packages/MixedModels/dn0WY/src/MixedModels.jl*

## Job Properties
* Time of benchmark: 2 Oct 2018 - 13:42
* Package commit: non gi
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

| ID                                                                                                                                                                                                  | time            | GC time    | memory          | allocations |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------:|----------------:|------------:|
| `["crossed", "Assay:1+A+B*C+(1|G)+(1|H)"]`                                                                                                                                                          |   2.943 ms (5%) |            | 449.23 KiB (1%) |        7821 |
| `["crossed", "Demand:1+U+V+W+X+(1|G)+(1|H)"]`                                                                                                                                                       |   2.775 ms (5%) |            | 386.13 KiB (1%) |        8827 |
| `["crossed", "InstEval:1+A*I+(1|G)+(1|H)"]`                                                                                                                                                         |    1.247 s (5%) | 114.131 ms | 234.50 MiB (1%) |       33070 |
| `["crossed", "InstEval:1+A+(1|G)+(1|H)+(1|I)"]`                                                                                                                                                     |    1.999 s (5%) |  12.898 ms | 187.33 MiB (1%) |       47246 |
| `["crossed", "Penicillin:1+(1|G)+(1|H)"]`                                                                                                                                                           |   2.697 ms (5%) |            | 350.83 KiB (1%) |        8064 |
| `["crossed", "ScotsSec:1+A+U+V+(1|G)+(1|H)"]`                                                                                                                                                       |   4.833 ms (5%) |            |   1.45 MiB (1%) |        9699 |
| `["crossed", "dialectNL:1+A+T+U+V+W+X+(1|G)+(1|H)+(1|I)"]`                                                                                                                                          | 416.892 ms (5%) |   6.731 ms |  95.20 MiB (1%) |       28416 |
| `["crossed", "egsingle:1+A+U+V+(1|G)+(1|H)"]`                                                                                                                                                       |  31.421 ms (5%) |   3.427 ms |  48.19 MiB (1%) |       16055 |
| `["crossed", "ml1m:1+(1|G)+(1|H)"]`                                                                                                                                                                 |   36.714 s (5%) | 225.872 ms | 323.09 MiB (1%) |     2045434 |
| `["crossed", "paulsim:1+S+T+U+(1|H)+(1|G)"]`                                                                                                                                                        |  14.097 ms (5%) |            |   4.41 MiB (1%) |       10208 |
| `["crossedvector", "bs10:1+U+V+W+((1+U+V+W)|G)+((1+U+V+W)|H)"]`                                                                                                                                     | 165.171 ms (5%) |   3.149 ms |  25.47 MiB (1%) |      806498 |
| `["crossedvector", "d3:1+U+((1+U)|G)+((1+U)|H)+((1+U)|I)"]`                                                                                                                                         |   49.023 s (5%) |    1.766 s |   7.51 GiB (1%) |   301762163 |
| `["crossedvector", "d3:1+U+(1|G)+(1|H)+(1|I)"]`                                                                                                                                                     | 299.348 ms (5%) | 117.923 ms | 371.75 MiB (1%) |       43708 |
| `["crossedvector", "gb12:1+S+T+U+V+W+X+Z+((1+S+U+W)|G)+((1+S+T+V)|H)"]`                                                                                                                             | 134.101 ms (5%) |            |  15.88 MiB (1%) |      537616 |
| `["crossedvector", "kb07:1+S+T+U+V+W+X+Z+((1+S+T+U+V+W+X+Z)|G)+((1+S+T+U+V+W+X+Z)|H)"]`                                                                                                             |    3.488 s (5%) |  16.508 ms | 124.58 MiB (1%) |     4319046 |
| `["crossedvector", "kb07:1+S+T+U+V+W+X+Z+(1|G)+((0+S)|G)+((0+T)|G)+((0+U)|G)+((0+V)|G)+((0+W)|G)+((0+X)|G)+((0+Z)|G)+(1|H)+((0+S)|H)+((0+T)|H)+((0+U)|H)+((0+V)|H)+((0+W)|H)+((0+X)|H)+((0+Z)|H)"]` | 493.390 ms (5%) |   7.953 ms |  70.86 MiB (1%) |     3239747 |
| `["nested", "Animal:1+(1|G)+(1|H)"]`                                                                                                                                                                |   1.261 ms (5%) |            | 178.91 KiB (1%) |        3819 |
| `["nested", "Chem97:1+(1|G)+(1|H)"]`                                                                                                                                                                |  58.460 ms (5%) |   6.975 ms |  93.76 MiB (1%) |       19565 |
| `["nested", "Chem97:1+U+(1|G)+(1|H)"]`                                                                                                                                                              |  59.353 ms (5%) |   7.019 ms |  94.54 MiB (1%) |       19736 |
| `["nested", "Genetics:1+A+(1|G)+(1|H)"]`                                                                                                                                                            |   2.062 ms (5%) |            | 317.86 KiB (1%) |        6566 |
| `["nested", "Pastes:1+(1|G)+(1|H)"]`                                                                                                                                                                |   2.298 ms (5%) |            | 326.86 KiB (1%) |        7028 |
| `["nested", "Semi2:1+A+(1|G)+(1|H)"]`                                                                                                                                                               |   2.309 ms (5%) |            | 352.11 KiB (1%) |        7236 |
| `["simplescalar", "Alfalfa:1+A*B+(1|G)"]`                                                                                                                                                           |   1.210 ms (5%) |            | 208.80 KiB (1%) |        3528 |
| `["simplescalar", "Alfalfa:1+A+B+(1|G)"]`                                                                                                                                                           |   1.021 ms (5%) |            | 168.47 KiB (1%) |        2901 |
| `["simplescalar", "AvgDailyGain:1+A*U+(1|G)"]`                                                                                                                                                      |   1.287 ms (5%) |            | 193.33 KiB (1%) |        3811 |
| `["simplescalar", "AvgDailyGain:1+A+U+(1|G)"]`                                                                                                                                                      |   1.144 ms (5%) |            | 169.59 KiB (1%) |        3294 |
| `["simplescalar", "BIB:1+A*U+(1|G)"]`                                                                                                                                                               |   1.574 ms (5%) |            | 222.20 KiB (1%) |        4738 |
| `["simplescalar", "BIB:1+A+U+(1|G)"]`                                                                                                                                                               |   1.171 ms (5%) |            | 171.31 KiB (1%) |        3384 |
| `["simplescalar", "Bond:1+A+(1|G)"]`                                                                                                                                                                | 958.770 μs (5%) |            | 141.25 KiB (1%) |        2615 |
| `["simplescalar", "Cultivation:1+A*B+(1|G)"]`                                                                                                                                                       |   1.089 ms (5%) |            | 173.38 KiB (1%) |        3298 |
| `["simplescalar", "Cultivation:1+A+(1|G)"]`                                                                                                                                                         |   1.138 ms (5%) |            | 162.14 KiB (1%) |        3254 |
| `["simplescalar", "Cultivation:1+A+B+(1|G)"]`                                                                                                                                                       |   1.147 ms (5%) |            | 173.47 KiB (1%) |        3433 |
| `["simplescalar", "Dyestuff2:1+(1|G)"]`                                                                                                                                                             | 830.840 μs (5%) |            | 105.20 KiB (1%) |        2225 |
| `["simplescalar", "Dyestuff:1+(1|G)"]`                                                                                                                                                              | 974.091 μs (5%) |            | 120.86 KiB (1%) |        2692 |
| `["simplescalar", "Exam:1+A*U+B+(1|G)"]`                                                                                                                                                            |   2.250 ms (5%) |            |   1.17 MiB (1%) |        4662 |
| `["simplescalar", "Exam:1+A+B+U+(1|G)"]`                                                                                                                                                            |   2.133 ms (5%) |            |   1.03 MiB (1%) |        4325 |
| `["simplescalar", "Gasoline:1+U+(1|G)"]`                                                                                                                                                            |   1.164 ms (5%) |            | 162.03 KiB (1%) |        3294 |
| `["simplescalar", "Hsb82:1+A+B+C+U+(1|G)"]`                                                                                                                                                         |   3.048 ms (5%) |            |   2.12 MiB (1%) |        4611 |
| `["simplescalar", "IncBlk:1+A+U+V+W+Z+(1|G)"]`                                                                                                                                                      |   1.226 ms (5%) |            | 208.83 KiB (1%) |        4135 |
| `["simplescalar", "Mississippi:1+A+(1|G)"]`                                                                                                                                                         | 980.968 μs (5%) |            | 145.75 KiB (1%) |        2704 |
| `["simplescalar", "PBIB:1+A+(1|G)"]`                                                                                                                                                                |   1.509 ms (5%) |            | 234.47 KiB (1%) |        3881 |
| `["simplescalar", "Rail:1+(1|G)"]`                                                                                                                                                                  |   1.251 ms (5%) |            | 151.34 KiB (1%) |        3622 |
| `["simplescalar", "Semiconductor:1+A*B+(1|G)"]`                                                                                                                                                     |   1.313 ms (5%) |            | 222.95 KiB (1%) |        3674 |
| `["simplescalar", "TeachingII:1+A+T+U+V+W+X+Z+(1|G)"]`                                                                                                                                              |   1.483 ms (5%) |            | 284.53 KiB (1%) |        5472 |
| `["simplescalar", "cake:1+A*B+(1|G)"]`                                                                                                                                                              |   1.606 ms (5%) |            | 412.83 KiB (1%) |        3666 |
| `["simplescalar", "ergoStool:1+A+(1|G)"]`                                                                                                                                                           |   1.057 ms (5%) |            | 155.59 KiB (1%) |        2913 |
| `["singlevector", "Early:1+U+U&A+((1+U)|G)"]`                                                                                                                                                       |  20.373 ms (5%) |            |   3.47 MiB (1%) |       80473 |
| `["singlevector", "HR:1+A*U+V+((1+U)|G)"]`                                                                                                                                                          |   5.183 ms (5%) |            | 915.00 KiB (1%) |       27962 |
| `["singlevector", "Oxboys:1+U+((1+U)|G)"]`                                                                                                                                                          |  13.207 ms (5%) |            |   1.93 MiB (1%) |       51919 |
| `["singlevector", "SIMS:1+U+((1+U)|G)"]`                                                                                                                                                            |  61.675 ms (5%) |            |  12.86 MiB (1%) |      394095 |
| `["singlevector", "WWheat:1+U+((1+U)|G)"]`                                                                                                                                                          |   7.311 ms (5%) |            | 902.31 KiB (1%) |       24071 |
| `["singlevector", "Weights:1+A*U+((1+U)|G)"]`                                                                                                                                                       |  18.303 ms (5%) |            |   3.20 MiB (1%) |       92915 |
| `["singlevector", "sleepstudy:1+U+((1+U)|G)"]`                                                                                                                                                      |   4.829 ms (5%) |            | 797.48 KiB (1%) |       23820 |
| `["singlevector", "sleepstudy:1+U+(1|G)+((0+U)|G)"]`                                                                                                                                                |   3.219 ms (5%) |            | 605.13 KiB (1%) |       19180 |

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
  uname: Linux 4.15.0-36-generic #39-Ubuntu SMP Mon Sep 24 16:19:09 UTC 2018 x86_64 x86_64
  CPU: Intel(R) Core(TM) i5-3570 CPU @ 3.40GHz: 
              speed         user         nice          sys         idle          irq
       #1  1690 MHz     140498 s        134 s      18382 s    1495130 s          0 s
       #2  2513 MHz     131505 s         16 s      18277 s    1504212 s          0 s
       #3  1900 MHz     145131 s        581 s      18892 s    1485409 s          0 s
       #4  1682 MHz     190751 s         38 s      17941 s    1445446 s          0 s
       
  Memory: 15.554645538330078 GB (10502.1171875 MB free)
  Uptime: 16578.0 sec
  Load Avg:  1.4091796875  2.07080078125  1.63037109375
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.0 (ORCJIT, ivybridge)
```
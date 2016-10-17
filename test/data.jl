using Compat, DataFrames
## Dyestuff data from lme4
const ds = DataFrame(Yield = [1545.,1440.,1440.,1520.,1580.,1540.,1555.,1490.,1560.,1495.,
                              1595.,1550.,1605.,1510.,1560.,1445.,1440.,1595.,1465.,1545.,
                              1595.,1630.,1515.,1635.,1625.,1520.,1455.,1450.,1480.,1445.],
                     Batch = categorical(Compat.repeat('A' : 'F', inner = 5)))

## Dyestuff2 data from lme4
const ds2 = DataFrame(Yield = [7.298,3.846,2.434,9.566,7.99,5.22,6.556,0.608,11.788,-0.892,
                               0.11, 10.386,13.434,5.51,8.166,2.212,4.852,7.092,9.288,4.98,
                               0.282,9.014,4.458,9.446,7.198,1.722,4.782,8.106,0.758,3.758],
                      Batch = categorical(Compat.repeat('A' : 'F', inner = 5)))

## sleepstudy data from lme4

const slp = DataFrame(Reaction =
                      [249.56,258.7047,250.8006,321.4398,356.8519,414.6901,382.2038,
                       290.1486,430.5853,466.3535,222.7339,205.2658,202.9778,204.707,
                       207.7161,215.9618,213.6303,217.7272,224.2957,237.3142,199.0539,
                       194.3322,234.32,232.8416,229.3074,220.4579,235.4208,255.7511,
                       261.0125,247.5153,321.5426,300.4002,283.8565,285.133,285.7973,
                       297.5855,280.2396,318.2613,305.3495,354.0487,287.6079,285.0,
                       301.8206,320.1153,316.2773,293.3187,290.075,334.8177,293.7469,
                       371.5811,234.8606,242.8118,272.9613,309.7688,317.4629,309.9976,
                       454.1619,346.8311,330.3003,253.8644,283.8424,289.555,276.7693,
                       299.8097,297.171,338.1665,332.0265,348.8399,333.36,362.0428,
                       265.4731,276.2012,243.3647,254.6723,279.0244,284.1912,305.5248,
                       331.5229,335.7469,377.299,241.6083,273.9472,254.4907,270.8021,
                       251.4519,254.6362,245.4523,235.311,235.7541,237.2466,312.3666,
                       313.8058,291.6112,346.1222,365.7324,391.8385,404.2601,416.6923,
                       455.8643,458.9167,236.1032,230.3167,238.9256,254.922,250.7103,
                       269.7744,281.5648,308.102,336.2806,351.6451,256.2968,243.4543,
                       256.2046,255.5271,268.9165,329.7247,379.4445,362.9184,394.4872,
                       389.0527,250.5265,300.0576,269.8939,280.5891,271.8274,304.6336,
                       287.7466,266.5955,321.5418,347.5655,221.6771,298.1939,326.8785,
                       346.8555,348.7402,352.8287,354.4266,360.4326,375.6406,388.5417,
                       271.9235,268.4369,257.2424,277.6566,314.8222,317.2135,298.1353,
                       348.1229,340.28,366.5131,225.264,234.5235,238.9008,240.473,
                       267.5373,344.1937,281.1481,347.5855,365.163,372.2288,269.8804,
                       272.4428,277.8989,281.7895,279.1705,284.512,259.2658,304.6306,
                       350.7807,369.4692,269.4117,273.474,297.5968,310.6316,287.1726,
                       329.6076,334.4818,343.2199,369.1417,364.1236],
                      Days = Compat.repeat(0 : 9, outer = 18),
                      Subject = categorical(Compat.repeat(1 : 18, inner = 10)))

const bb = Compat.repeat('A' : 'J', inner = 6)
const cc = Compat.repeat('a' : 'c', inner = 2, outer = 10)

## Pastes data from the lme4 package
const psts = DataFrame(Strength = [62.8,62.6,60.1,62.3,62.7,63.1,60.0,61.4,57.5,56.9,61.1,58.9,
                                   58.7,57.5,63.9,63.1,65.4,63.7,57.1,56.4,56.9,58.6,64.7,64.5,
                                   55.1,55.1,54.7,54.2,58.8,57.5,63.4,64.9,59.3,58.1,60.5,60.0,
                                   62.5,62.6,61.0,58.7,56.9,57.7,59.2,59.4,65.2,66.0,64.8,64.1,
                                   54.8,54.8,64.0,64.0,57.7,56.8,58.3,59.3,59.2,59.2,58.9,56.6],
                       Batch = categorical(bb),
                       Cask = categorical(cc),
                       Sample = categorical(Compat.ASCIIString[string(b, c) for (b,c) in zip(bb,cc)]))

## Penicillin data from the lme4 package
const pen = DataFrame(Diameter = [27,23,26,23,23,21,27,23,26,23,23,21,25,21,25,24,24,20,26,23,25,
                                  23,23,20,25,22,26,22,23,20,24,22,25,23,22,19,24,20,23,21,22,19,
                                  26,22,26,24,24,21,24,21,24,22,22,20,24,21,24,23,22,19,26,23,26,
                                  24,24,21,25,22,26,24,24,20,26,24,26,24,25,22,26,23,26,23,23,20,
                                  26,23,25,24,24,22,25,22,25,23,23,20,25,21,24,23,23,20,25,22,24,
                                  23,23,19,24,21,23,21,21,19,26,23,26,24,24,21,25,21,24,22,22,18,
                                  25,22,25,22,22,20,24,21,24,22,24,19,24,21,24,22,21,18],
                      Plate = categorical(Compat.repeat('a' : 'x', inner = 6)),
                      Sample = categorical(Compat.repeat(collect('A' : 'F'), outer = 24)))

## InstEval data from the lme4 package

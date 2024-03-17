PDP1 = 0.01
PDP2 = 0.03
PDP3 = 0.02

P1 = 0.3
P2 = 0.2
P3 = 0.5

ProbabilityP1 = (PDP1*P1)/((PDP1*P1) + (PDP2*P2) + (PDP3*P3))
ProbabilityP2 = (PDP2*P2)/((PDP1*P1) + (PDP2*P2) + (PDP3*P3))
ProbabilityP3 = (PDP3*P3)/((PDP1*P1) + (PDP2*P2) + (PDP3*P3))

print(ProbabilityP1)
print(ProbabilityP2)
print(ProbabilityP3)

print("Part 3 will be responsible.")
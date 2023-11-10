def F_s(SL, threshold, AR, OR):
    pe1 = sum([(max(threshold[c] - SL[c] * 100, 0)) ** 2 + (AR[c] * 100) ** 2 for c in SL.keys()])
    pe2 = sum([(max(OR[s]*100-95, 0))**2 for s in OR.keys()])
    return pe1+pe2

if __name__ == "__main__":
    pass
from collections import defaultdict

c = defaultdict(int)
for line in open("hist.txt", "r", encoding="utf-16").readlines():
    c[line.strip()] += 1

for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v}")


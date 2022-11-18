import sys

with open(sys.argv[1]) as f:
    f.readline()  # header
    res = [l.strip() for l in f]

with open(sys.argv[2]) as f:
    groundtruth = [l.split('\t')[1].strip() for l in f]

n = len(groundtruth)
c = sum(i == j for i, j in zip(groundtruth, res))
print(c, "samples are correct out of", n, "with accuracy", c / n)

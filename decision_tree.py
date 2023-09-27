import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

observations = []
f = open(args.filename)
for line in f.readlines():
    # print(line.strip())
    observation = tuple(float(e) for e in line.strip().split(sep=" "))
    print(observation)
    observations.append(observation)

num_features = 2
runs_sorted = []
splits = []
for j in range(num_features):
    run_j = sorted(observations, key= lambda x: x[j])
    print(run_j)
    runs_sorted.append(run_j)

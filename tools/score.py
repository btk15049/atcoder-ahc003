import sys
import glob

assert len(sys.argv) == 2
files = glob.glob(f'{sys.argv[1]}/*')

a = []
for file in files:
    with open(file) as f:
        s = f.read().rstrip()
        if len(s) > 0:
            a.append(int(s.rstrip()))


print(f'average of {len(a)}: {sum(a) / len(a)}')

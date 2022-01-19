a, b = map(int, input().split())
d = input()

fibs = [1, 2]

res = []
if d != '1' and a == 1:
    res.append('1')

if d != '2' and a <= 2:
    res.append('2')

while fibs[-1] + fibs[-2] <= b:
    fibs.append(fibs[-1] + fibs[-2])
    if fibs[-1] >= a and d not in str(fibs[-1]):
        res.append(str(fibs[-1]))

print(len(res))
print(' '.join(res))

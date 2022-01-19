n, d = map(int, input().split())
s = input()

current_len = 1
counts = [set()]

for i, c in enumerate(s):
    for j in range(i + 1):
        l = i - j + 1
        if l > len(counts) - 1:
            counts.append(set())
        counts[l].add(s[j:i + 1])
        if len(counts[l]) == d ** l and len(counts[l + 1]) < d ** (l + 1):
            current_len = l + 1
    print(current_len, end=' ')

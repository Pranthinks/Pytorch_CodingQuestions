
val = int(input())

def LIS(nums):
    for i in range(len(nums)):
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp1[i] = max(dp1[i], 1 + dp1[j])
    

def LDS(nums):
    for i in range(len(nums)-1, -1, -1):
        for j in range(i+1, len(nums)):
            if nums[i] > nums[j]:
                dp2[i] = max(dp2[i], 1 + dp2[j])
    return max(dp2)


for i in range(val):
    res = []
    n = int(input())
    res.extend(map(int, input().split()))
    dp1 = [1]* len(res)
    dp2 = [1]* len(res)
    if n == 1:
        print(0)
        exit()
    LIS(res)
    LDS(res)
    max_val = 0
    for i in range(len(res)):
        if dp1[i] > 1 and dp2[i] > 1:
            max_val = max(max_val, dp1[i]+dp2[i]-1)
    print(max_val)
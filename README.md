# hot100

> 题单: https://leetcode.cn/studyplan/top-100-liked

## 哈希

### [1. 两数之和](https://leetcode.cn/problems/two-sum/description)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mp = {}
        for i, num in enumerate(nums):
            if target - num in mp:
                return [mp[target - num], i]
            mp[num] = i
        return []
```

### [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/description)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            mp[key].append(s)
        return list(mp.values())
```

### [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description)

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        res = 0
        for num in nums:
            if num - 1 in nums:
                continue
            tmp = 1
            while num + 1 in nums:
                tmp += 1
                num += 1
            res = max(res, tmp)
        return res
```

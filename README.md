# hot100

> 题单: https://leetcode.cn/studyplan/top-100-liked

## 哈希

### [1. 两数之和 (Two Sum)](https://leetcode.cn/problems/two-sum/description)

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

### [49. 字母异位词分组 (Group Anagrams)](https://leetcode.cn/problems/group-anagrams/description)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            mp[key].append(s)
        return list(mp.values())
```

### [128. 最长连续序列 (Longest Consecutive Sequence)](https://leetcode.cn/problems/longest-consecutive-sequence/description)

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

## 双指针

### [283. 移动零 (Move Zeroes)](https://leetcode.cn/problems/move-zeroes/description)

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
```

### [11. 盛最多水的容器 (Container With Most Water)](https://leetcode.cn/problems/container-with-most-water/description)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        res = 0
        while left < right:
            h = min(height[left], height[right])
            w = right - left
            res = max(res, h * w)
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return res
```

### [15. 三数之和 (3Sum)](https://leetcode.cn/problems/3sum/description)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for first in range(n - 2):
            if first > 0 and nums[first - 1] == nums[first]:
                continue
            third = n - 1
            target = - nums[first]
            for second in range(first + 1, n - 1):
                if second > first + 1 and nums[second - 1] == nums[second]:
                    continue
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                if second == third:
                    break
                elif nums[second] + nums[third] == target:
                    res.append([nums[first], nums[second], nums[third]])
        return res
```

### [42. 接雨水 (Trapping Rain Water)](https://leetcode.cn/problems/trapping-rain-water/description)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_l, max_r = 0, 0
        res = 0
        while left < right:
            max_l = max(max_l, height[left])
            max_r = max(max_r, height[right])
            if height[left] <= height[right]:
                res += (max_l - height[left])
                left += 1
            else:
                res += (max_r - height[right])
                right -= 1
        return res
```

## 滑动窗口

### [3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        window = set()
        res = 0
        left = 0
        for right in range(len(s)):
            while s[right] in window:
                window.remove(s[left])
                left += 1
            window.add(s[right])
            res = max(res, right - left + 1)
        return res
```

### [438. 找到字符串中所有字母异位词 (Find All Anagrams in a String)](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description)

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        window = Counter(p)
        left = 0
        res = []
        for right in range(len(s)):
            window[s[right]] -= 1
            while window[s[right]] < 0:
                window[s[left]] += 1
                left += 1
            if right - left + 1 == len(p):
                res.append(left)
        return res
```

## 子串

### [560. 和为 K 的子数组 (Subarray Sum Equals K)](https://leetcode.cn/problems/subarray-sum-equals-k/description)
### [239. 滑动窗口最大值 (Sliding Window Maximum)](https://leetcode.cn/problems/sliding-window-maximum/description)
### [76. 最小覆盖子串 (Minimum Window Substring)](https://leetcode.cn/problems/minimum-window-substring/description)
## 普通数组
### [53. 最大子数组和 (Maximum Subarray)](https://leetcode.cn/problems/maximum-subarray/description)
### [56. 合并区间 (Merge Intervals)](https://leetcode.cn/problems/merge-intervals/description)
### [189. 轮转数组 (Rotate Array)](https://leetcode.cn/problems/rotate-array/description)
### [238. 除自身以外数组的乘积 (Product of Array Except Self)](https://leetcode.cn/problems/product-of-array-except-self/description)
### [41. 缺失的第一个正数 (First Missing Positive)](https://leetcode.cn/problems/first-missing-positive/description)
## 矩阵
### [73. 矩阵置零 (Set Matrix Zeroes)](https://leetcode.cn/problems/set-matrix-zeroes/description)
### [54. 螺旋矩阵 (Spiral Matrix)](https://leetcode.cn/problems/spiral-matrix/description)
### [48. 旋转图像 (Rotate Image)](https://leetcode.cn/problems/rotate-image/description)
### [240. 搜索二维矩阵 II (Search a 2D Matrix II)](https://leetcode.cn/problems/search-a-2d-matrix-ii/description)
## 链表
### [160. 相交链表 (Intersection of Two Linked Lists)](https://leetcode.cn/problems/intersection-of-two-linked-lists/description)
### [206. 反转链表 (Reverse Linked List)](https://leetcode.cn/problems/reverse-linked-list/description)
### [234. 回文链表 (Palindrome Linked List)](https://leetcode.cn/problems/palindrome-linked-list/description)
### [141. 环形链表 (Linked List Cycle)](https://leetcode.cn/problems/linked-list-cycle/description)
### [142. 环形链表 II (Linked List Cycle II)](https://leetcode.cn/problems/linked-list-cycle-ii/description)
### [21. 合并两个有序链表 (Merge Two Sorted Lists)](https://leetcode.cn/problems/merge-two-sorted-lists/description)
### [2. 两数相加 (Add Two Numbers)](https://leetcode.cn/problems/add-two-numbers/description)
### [19. 删除链表的倒数第 N 个结点 (Remove Nth Node From End of List)](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description)
### [24. 两两交换链表中的节点 (Swap Nodes in Pairs)](https://leetcode.cn/problems/swap-nodes-in-pairs/description)
### [25. K 个一组翻转链表 (Reverse Nodes in k-Group)](https://leetcode.cn/problems/reverse-nodes-in-k-group/description)
### [138. 随机链表的复制 (Copy List with Random Pointer)](https://leetcode.cn/problems/copy-list-with-random-pointer/description)
### [148. 排序链表 (Sort List)](https://leetcode.cn/problems/sort-list/description)
### [23. 合并 K 个升序链表 (Merge k Sorted Lists)](https://leetcode.cn/problems/merge-k-sorted-lists/description)
### [146. LRU 缓存 (LRU Cache)](https://leetcode.cn/problems/lru-cache/description)
## 二叉树
### [94. 二叉树的中序遍历 (Binary Tree Inorder Traversal)](https://leetcode.cn/problems/binary-tree-inorder-traversal/description)
### [104. 二叉树的最大深度 (Maximum Depth of Binary Tree)](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description)
### [226. 翻转二叉树 (Invert Binary Tree)](https://leetcode.cn/problems/invert-binary-tree/description)
### [101. 对称二叉树 (Symmetric Tree)](https://leetcode.cn/problems/symmetric-tree/description)
### [543. 二叉树的直径 (Diameter of Binary Tree)](https://leetcode.cn/problems/diameter-of-binary-tree/description)
### [102. 二叉树的层序遍历 (Binary Tree Level Order Traversal)](https://leetcode.cn/problems/binary-tree-level-order-traversal/description)
### [108. 将有序数组转换为二叉搜索树 (Convert Sorted Array to Binary Search Tree)](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description)
### [98. 验证二叉搜索树 (Validate Binary Search Tree)](https://leetcode.cn/problems/validate-binary-search-tree/description)
### [230. 二叉搜索树中第 K 小的元素 (Kth Smallest Element in a BST)](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description)
### [199. 二叉树的右视图 (Binary Tree Right Side View)](https://leetcode.cn/problems/binary-tree-right-side-view/description)
### [114. 二叉树展开为链表 (Flatten Binary Tree to Linked List)](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description)
### [105. 从前序与中序遍历序列构造二叉树 (Construct Binary Tree from Preorder and Inorder Traversal)](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description)
### [437. 路径总和 III (Path Sum III)](https://leetcode.cn/problems/path-sum-iii/description)
### [236. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree)](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description)
### [124. 二叉树中的最大路径和 (Binary Tree Maximum Path Sum)](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description)
## 图论
### [200. 岛屿数量 (Number of Islands)](https://leetcode.cn/problems/number-of-islands/description)
### [994. 腐烂的橘子 (Rotting Oranges)](https://leetcode.cn/problems/rotting-oranges/description)
### [207. 课程表 (Course Schedule)](https://leetcode.cn/problems/course-schedule/description)
### [208. 实现 Trie (前缀树) (Implement Trie (Prefix Tree))](https://leetcode.cn/problems/implement-trie-prefix-tree/description)
## 回溯
### [46. 全排列 (Permutations)](https://leetcode.cn/problems/permutations/description)
### [78. 子集 (Subsets)](https://leetcode.cn/problems/subsets/description)
### [17. 电话号码的字母组合 (Letter Combinations of a Phone Number)](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description)
### [39. 组合总和 (Combination Sum)](https://leetcode.cn/problems/combination-sum/description)
### [22. 括号生成 (Generate Parentheses)](https://leetcode.cn/problems/generate-parentheses/description)
### [79. 单词搜索 (Word Search)](https://leetcode.cn/problems/word-search/description)
### [131. 分割回文串 (Palindrome Partitioning)](https://leetcode.cn/problems/palindrome-partitioning/description)
### [51. N 皇后 (N-Queens)](https://leetcode.cn/problems/n-queens/description)
## 二分查找
### [35. 搜索插入位置 (Search Insert Position)](https://leetcode.cn/problems/search-insert-position/description)
### [74. 搜索二维矩阵 (Search a 2D Matrix)](https://leetcode.cn/problems/search-a-2d-matrix/description)
### [34. 在排序数组中查找元素的第一个和最后一个位置 (Find First and Last Position of Element in Sorted Array)](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description)
### [33. 搜索旋转排序数组 (Search in Rotated Sorted Array)](https://leetcode.cn/problems/search-in-rotated-sorted-array/description)
### [153. 寻找旋转排序数组中的最小值 (Find Minimum in Rotated Sorted Array)](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description)
### [4. 寻找两个正序数组的中位数 (Median of Two Sorted Arrays)](https://leetcode.cn/problems/median-of-two-sorted-arrays/description)
## 栈
### [20. 有效的括号 (Valid Parentheses)](https://leetcode.cn/problems/valid-parentheses/description)
### [155. 最小栈 (Min Stack)](https://leetcode.cn/problems/min-stack/description)
### [394. 字符串解码 (Decode String)](https://leetcode.cn/problems/decode-string/description)
### [739. 每日温度 (Daily Temperatures)](https://leetcode.cn/problems/daily-temperatures/description)
### [84. 柱状图中最大的矩形 (Largest Rectangle in Histogram)](https://leetcode.cn/problems/largest-rectangle-in-histogram/description)
## 堆
### [215. 数组中的第K个最大元素 (Kth Largest Element in an Array)](https://leetcode.cn/problems/kth-largest-element-in-an-array/description)
### [347. 前 K 个高频元素 (Top K Frequent Elements)](https://leetcode.cn/problems/top-k-frequent-elements/description)
### [295. 数据流的中位数 (Find Median from Data Stream)](https://leetcode.cn/problems/find-median-from-data-stream/description)
## 贪心算法
### [121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description)
### [55. 跳跃游戏 (Jump Game)](https://leetcode.cn/problems/jump-game/description)
### [45. 跳跃游戏 II (Jump Game II)](https://leetcode.cn/problems/jump-game-ii/description)
### [763. 划分字母区间 (Partition Labels)](https://leetcode.cn/problems/partition-labels/description)
## 动态规划
### [70. 爬楼梯 (Climbing Stairs)](https://leetcode.cn/problems/climbing-stairs/description)
### [118. 杨辉三角 (Pascal's Triangle)](https://leetcode.cn/problems/pascals-triangle/description)
### [198. 打家劫舍 (House Robber)](https://leetcode.cn/problems/house-robber/description)
### [279. 完全平方数 (Perfect Squares)](https://leetcode.cn/problems/perfect-squares/description)
### [322. 零钱兑换 (Coin Change)](https://leetcode.cn/problems/coin-change/description)
### [139. 单词拆分 (Word Break)](https://leetcode.cn/problems/word-break/description)
### [300. 最长递增子序列 (Longest Increasing Subsequence)](https://leetcode.cn/problems/longest-increasing-subsequence/description)
### [152. 乘积最大子数组 (Maximum Product Subarray)](https://leetcode.cn/problems/maximum-product-subarray/description)
### [416. 分割等和子集 (Partition Equal Subset Sum)](https://leetcode.cn/problems/partition-equal-subset-sum/description)
### [32. 最长有效括号 (Longest Valid Parentheses)](https://leetcode.cn/problems/longest-valid-parentheses/description)
## 多维动态规划
### [62. 不同路径 (Unique Paths)](https://leetcode.cn/problems/unique-paths/description)
### [64. 最小路径和 (Minimum Path Sum)](https://leetcode.cn/problems/minimum-path-sum/description)
### [5. 最长回文子串 (Longest Palindromic Substring)](https://leetcode.cn/problems/longest-palindromic-substring/description)
### [1143. 最长公共子序列 (Longest Common Subsequence)](https://leetcode.cn/problems/longest-common-subsequence/description)
### [72. 编辑距离 (Edit Distance)](https://leetcode.cn/problems/edit-distance/description)
## 技巧
### [136. 只出现一次的数字 (Single Number)](https://leetcode.cn/problems/single-number/description)
### [169. 多数元素 (Majority Element)](https://leetcode.cn/problems/majority-element/description)
### [75. 颜色分类 (Sort Colors)](https://leetcode.cn/problems/sort-colors/description)
### [31. 下一个排列 (Next Permutation)](https://leetcode.cn/problems/next-permutation/description)
### [287. 寻找重复数 (Find the Duplicate Number)](https://leetcode.cn/problems/find-the-duplicate-number/description)

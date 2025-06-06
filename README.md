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
        ml, mr = 0, 0
        res = 0
        while left < right:
            ml = max(ml, height[left])
            mr = max(mr, height[right])
            if height[left] <= height[right]:
                res += ml - height[left]
                left += 1
            else:
                res += mr - height[right]
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

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = defaultdict(int)
        cnt[0] = 1
        res = 0
        pre = 0
        for num in nums:
            pre += num
            res += cnt[pre - k]
            cnt[pre] += 1
        return res
```

### [239. 滑动窗口最大值 (Sliding Window Maximum)](https://leetcode.cn/problems/sliding-window-maximum/description)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()
        res = []
        for i in range(len(nums)):
            while q and nums[q[-1]] < nums[i]:
                q.pop()
            q.append(i)
            if i - q[0] + 1 > k:
                q.popleft()
            if i >= k - 1:
                res.append(nums[q[0]])
        return res
```

### [76. 最小覆盖子串 (Minimum Window Substring)](https://leetcode.cn/problems/minimum-window-substring/description)

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        cnt = Counter(t)
        less = len(cnt)
        ml, mr = -1, len(s)
        left = 0
        for right in range(len(s)):
            cnt[s[right]] -= 1
            if cnt[s[right]] == 0:
                less -= 1
            while less == 0:
                if right - left < mr - ml:
                    ml, mr = left, right
                if cnt[s[left]] == 0:
                    less += 1
                cnt[s[left]] += 1
                left += 1
        return s[ml: mr + 1] if ml != -1 else ''
```

## 普通数组

### [53. 最大子数组和 (Maximum Subarray)](https://leetcode.cn/problems/maximum-subarray/description)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = nums[0]
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i - 1] + nums[i])
            res = max(res, nums[i])
        return res
```

### [56. 合并区间 (Merge Intervals)](https://leetcode.cn/problems/merge-intervals/description)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = []
        for tmp in intervals:
            if res and res[-1][1] >= tmp[0]:
                res[-1][1] = max(res[-1][1], tmp[1])
            else:
                res.append(tmp)
        return res
```

### [189. 轮转数组 (Rotate Array)](https://leetcode.cn/problems/rotate-array/description)

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        nums[:] = nums[::-1]
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1]
```

### [238. 除自身以外数组的乘积 (Product of Array Except Self)](https://leetcode.cn/problems/product-of-array-except-self/description)

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        for i in range(n - 2, -1, -1):
            res[i] = res[i + 1] * nums[i + 1]
        pre = 1
        for i in range(n):
            res[i] *= pre
            pre *= nums[i]
        return res
```

### [41. 缺失的第一个正数 (First Missing Positive)](https://leetcode.cn/problems/first-missing-positive/description)

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            while 0 <= nums[i] - 1 <= len(nums) - 1 and nums[nums[i] - 1] != nums[i]:
                tmp = nums[i] - 1
                nums[tmp], nums[i] = nums[i], nums[tmp]

        for i in range(len(nums)):
            if nums[i] != i + 1:
                return i + 1
        return len(nums) + 1
```

## 矩阵

### [73. 矩阵置零 (Set Matrix Zeroes)](https://leetcode.cn/problems/set-matrix-zeroes/description)

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        row_flag = any(matrix[i][0] == 0 for i in range(len(matrix)))
        col_flag = any(matrix[0][j] == 0 for j in range(len(matrix[0])))

        for i in range(1, len(matrix)):
            for j in range(1, len((matrix[i]))):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0

        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if row_flag:
            for i in range(len(matrix)):
                matrix[i][0] = 0
        if col_flag:
            for j in range(len(matrix[0])):
                matrix[0][j] = 0
```

### [54. 螺旋矩阵 (Spiral Matrix)](https://leetcode.cn/problems/spiral-matrix/description)

```python
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        res = []
        i, j = 0, -1
        di = 0
        size = m * n
        while len(res) < size:
            dx, dy = DIRS[di]
            for _ in range(n):
                i += dx
                j += dy
                res.append(matrix[i][j])
            di = (di + 1) % 4
            n, m = m - 1, n
        return res
```

### [48. 旋转图像 (Rotate Image)](https://leetcode.cn/problems/rotate-image/description)

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(len(matrix)):
            matrix[i].reverse()
```

### [240. 搜索二维矩阵 II (Search a 2D Matrix II)](https://leetcode.cn/problems/search-a-2d-matrix-ii/description)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        i, j = 0, len(matrix[0]) - 1
        while i <= len(matrix) - 1 and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
        return False
```

## 链表

### [160. 相交链表 (Intersection of Two Linked Lists)](https://leetcode.cn/problems/intersection-of-two-linked-lists/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pA, pB = headA, headB
        while pA is not pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        return pA
```

### [206. 反转链表 (Reverse Linked List)](https://leetcode.cn/problems/reverse-linked-list/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        while head:
            tmp = head.next
            head.next = p.next
            p.next = head
            head = tmp
        return dummy.next
```

### [234. 回文链表 (Palindrome Linked List)](https://leetcode.cn/problems/palindrome-linked-list/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        def find_mid(head):
            p = dummy = ListNode(next=head)
            q = head
            while q and q.next:
                p = p.next
                q = q.next.next
            return p

        def reverse(head):
            p = dummy = ListNode()
            while head:
                tmp = head.next
                head.next = p.next
                p.next = head
                head = tmp
            return dummy.next

        p = find_mid(head)
        tmp = p.next
        p.next = None
        p = tmp
        p = reverse(p)
        while head and p and head.val == p.val:
            head = head.next
            p = p.next
        return not (head and p)
```

### [141. 环形链表 (Linked List Cycle)](https://leetcode.cn/problems/linked-list-cycle/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        p = q = head
        while q and q.next:
            p = p.next
            q = q.next.next
            if p is q:
                return True
        return False
```

### [142. 环形链表 II (Linked List Cycle II)](https://leetcode.cn/problems/linked-list-cycle-ii/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p = q = head
        while q and q.next:
            p = p.next
            q = q.next.next
            if p is q:
                break
        if not (q and q.next):
            return
        p = head
        while p is not q:
            p = p.next
            q = q.next
        return p
```

### [21. 合并两个有序链表 (Merge Two Sorted Lists)](https://leetcode.cn/problems/merge-two-sorted-lists/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                p.next = list1
                p = p.next
                list1 = list1.next
            else:
                p.next = list2
                p = p.next
                list2 = list2.next
        p.next = list1 or list2
        return dummy.next
```

### [2. 两数相加 (Add Two Numbers)](https://leetcode.cn/problems/add-two-numbers/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            p.next = ListNode(val=carry % 10)
            p = p.next
            carry //= 10
        return dummy.next
```

### [19. 删除链表的倒数第 N 个结点 (Remove Nth Node From End of List)](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        p = dummy = ListNode(next=head)
        q = head
        for _ in range(n):
            q = q.next
        while q:
            p = p.next
            q = q.next
        p.next = p.next.next
        return dummy.next
```

### [24. 两两交换链表中的节点 (Swap Nodes in Pairs)](https://leetcode.cn/problems/swap-nodes-in-pairs/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node0 = dummy = ListNode(next=head)
        node1 = head
        while node1 and node1.next:
            node2 = node1.next
            node3 = node2.next

            node0.next = node2
            node2.next = node1
            node1.next = node3

            node0 = node1
            node1 = node3
        return dummy.next
```

### [25. K 个一组翻转链表 (Reverse Nodes in k-Group)](https://leetcode.cn/problems/reverse-nodes-in-k-group/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        p = head
        for _ in range(k):
            if not p:
                return head
            p = p.next
        p, q = None, head
        for _ in range(k):
            tmp = q.next
            q.next = p
            p = q
            q = tmp
        head.next = self.reverseKGroup(q, k)
        return p
```

### [138. 随机链表的复制 (Copy List with Random Pointer)](https://leetcode.cn/problems/copy-list-with-random-pointer/description)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""


class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return
        mp = {}
        p = head
        while p:
            mp[p] = ListNode(p.val)
            p = p.next
        for k in mp:
            mp[k].next = mp.get(k.next)
            mp[k].random = mp.get(k.random)
        return mp[head]
```

### [148. 排序链表 (Sort List)](https://leetcode.cn/problems/sort-list/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not (head and head.next):
            return head
        p, q = head, head.next
        while q and q.next:
            p = p.next
            q = q.next.next
        tmp = p.next
        p.next = None
        p = tmp
        head, p = self.sortList(head), self.sortList(p)
        r = dummy = ListNode()
        while head and p:
            if head.val < p.val:
                r.next = ListNode(head.val)
                r = r.next
                head = head.next
            else:
                r.next = ListNode(p.val)
                r = r.next
                p = p.next
        r.next = head if head else p
        return dummy.next
```

### [23. 合并 K 个升序链表 (Merge k Sorted Lists)](https://leetcode.cn/problems/merge-k-sorted-lists/description)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


ListNode.__lt__ = lambda x, y: x.val < y.val
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        lists = [ l for l in lists if l ]
        heapify(lists)
        p = dummy = ListNode()
        while lists:
            node = heappop(lists)
            p.next = node
            p = p.next
            if node.next:
                heappush(lists, node.next)
        return dummy.next
```

### [146. LRU 缓存 (LRU Cache)](https://leetcode.cn/problems/lru-cache/description)

```python
class LRUCache(OrderedDict):

    def __init__(self, capacity: int):
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self:
            return -1
        self.move_to_end(key)
        return self[key]

    def put(self, key: int, value: int) -> None:
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)
```

## 二叉树

### [94. 二叉树的中序遍历 (Binary Tree Inorder Traversal)](https://leetcode.cn/problems/binary-tree-inorder-traversal/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)


class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        stack = [root]
        while stack:
            x = stack.pop()
            if isinstance(x, int):
                res.append(x)
            elif isinstance(x, TreeNode):
                stack.extend([x.right, x.val, x.left])
        return res
```

### [104. 二叉树的最大深度 (Maximum Depth of Binary Tree)](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        max_depth = 0
        while q:
            for _ in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            max_depth += 1
        return max_depth
```

### [226. 翻转二叉树 (Invert Binary Tree)](https://leetcode.cn/problems/invert-binary-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

### [101. 对称二叉树 (Symmetric Tree)](https://leetcode.cn/problems/symmetric-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def dfs(p, q):
            if not (p and q):
                return p is q
            return p.val == q.val and dfs(p.left, q.right) and dfs(p.right, q.left)
        return dfs(root, root)
```

### [543. 二叉树的直径 (Diameter of Binary Tree)](https://leetcode.cn/problems/diameter-of-binary-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.res = 0
        def dfs(root):
            if not root:
                return 0
            l, r = dfs(root.left), dfs(root.right)
            self.res = max(self.res, l + r)
            return max(l, r) + 1
        dfs(root)
        return self.res
```

### [102. 二叉树的层序遍历 (Binary Tree Level Order Traversal)](https://leetcode.cn/problems/binary-tree-level-order-traversal/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            tmp = []
            for _ in range(len(q)):
                node = q.popleft()
                tmp.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(tmp)
        return res
```

### [108. 将有序数组转换为二叉搜索树 (Convert Sorted Array to Binary Search Tree)](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return
        m = len(nums) // 2
        return TreeNode(
            val=nums[m],
            left=self.sortedArrayToBST(nums[:m]),
            right=self.sortedArrayToBST(nums[m + 1:])
        )
```

### [98. 验证二叉搜索树 (Validate Binary Search Tree)](https://leetcode.cn/problems/validate-binary-search-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(root, lower, upper):
            if not root:
                return True
            if not (lower < root.val < upper):
                return False
            return dfs(root.left, lower, root.val) and dfs(root.right, root.val, upper)
        return dfs(root, -inf, inf)
```

### [230. 二叉搜索树中第 K 小的元素 (Kth Smallest Element in a BST)](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        self.res = None
        def dfs(root):
            if not root:
                return
            dfs(root.left)

            self.k -= 1
            if self.k == 0:
                self.res = root.val

            dfs(root.right)
        dfs(root)
        return self.res
```

### [199. 二叉树的右视图 (Binary Tree Right Side View)](https://leetcode.cn/problems/binary-tree-right-side-view/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res = []
        q = deque([root])
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if i == n - 1:
                    res.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return res
```

### [114. 二叉树展开为链表 (Flatten Binary Tree to Linked List)](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        self.flatten(root.left)
        self.flatten(root.right)

        tmp = root.right
        root.right = root.left
        root.left = None
        p = root
        while p and p.right:
            p = p.right
        p.right = tmp
```

### [105. 从前序与中序遍历序列构造二叉树 (Construct Binary Tree from Preorder and Inorder Traversal)](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        mp = {val: i for i, val in enumerate(inorder)}
        def dfs(root, left, right):
            if left > right:
                return
            i = mp[preorder[root]]
            return TreeNode(
                val=preorder[root],
                left=dfs(root + 1, left, i - 1),
                right=dfs(i - left + 1 + root, i + 1, right)
            )
        return dfs(0, 0, len(inorder) - 1)
```

### [437. 路径总和 III (Path Sum III)](https://leetcode.cn/problems/path-sum-iii/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        cnt = defaultdict(int)
        cnt[0] = 1
        self.pre = 0
        self.res = 0

        def dfs(root):
            if not root:
                return

            self.pre += root.val
            self.res += cnt[self.pre - targetSum]
            cnt[self.pre] += 1

            dfs(root.left)
            dfs(root.right)

            cnt[self.pre] -= 1
            self.pre -= root.val
        dfs(root)
        return self.res
```

### [236. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree)](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left or right
```

### [124. 二叉树中的最大路径和 (Binary Tree Maximum Path Sum)](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = -inf
        def dfs(root):
            if not root:
                return 0
            l, r = dfs(root.left), dfs(root.right)
            self.res = max(self.res, l + r + root.val)
            return max(max(l, r) + root.val, 0)
        dfs(root)
        return self.res
```

## 图论

### [200. 岛屿数量 (Number of Islands)](https://leetcode.cn/problems/number-of-islands/description)

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(i, j):
            if not (
                0 <= i <= len(grid) - 1
                and 0 <= j <= len(grid[0]) - 1
            ) or grid[i][j] != '1':
                return
            grid[i][j] = '2'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res
```

### [994. 腐烂的橘子 (Rotting Oranges)](https://leetcode.cn/problems/rotting-oranges/description)

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        fresh = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    q.append((i, j))

        rounder = 0
        while fresh and q:
            rounder += 1
            for _ in range(len(q)):
                i, j = q.popleft()
                for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if (
                        0 <= x <= len(grid) - 1
                        and 0 <= y <= len(grid[0]) - 1
                        and grid[x][y] == 1
                    ):
                        fresh -= 1
                        grid[x][y] = 2
                        q.append((x, y))
        return rounder if not fresh else -1
```

### [207. 课程表 (Course Schedule)](https://leetcode.cn/problems/course-schedule/description)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegs = [0] * numCourses
        adjs = defaultdict(list)
        for a, b in prerequisites:
            indegs[a] += 1
            adjs[b].append(a)

        q = deque()
        for i in range(len(indegs)):
            if indegs[i] == 0:
                q.append(i)

        while q:
            cur = q.popleft()
            numCourses -= 1
            for nxt in adjs[cur]:
                indegs[nxt] -= 1
                if indegs[nxt] == 0:
                    q.append(nxt)
        return numCourses == 0
```

### [208. 实现 Trie (前缀树) (Implement Trie (Prefix Tree))](https://leetcode.cn/problems/implement-trie-prefix-tree/description)

```python
class Trie(dict):

    def insert(self, word: str) -> None:
        for c in word:
            if c not in self:
                self[c] = Trie()
            self = self[c]
        self['$'] = Trie()

    def search(self, word: str) -> bool:
        for c in word:
            if c in self:
                self = self[c]
            else:
                return False
        return '$' in self

    def startsWith(self, prefix: str) -> bool:
        for c in prefix:
            if c in self:
                self = self[c]
            else:
                return False
        return True
```

## 回溯

### [46. 全排列 (Permutations)](https://leetcode.cn/problems/permutations/description)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                dfs(nums[:i] + nums[i + 1:], tmp + [nums[i]])
        dfs(nums, [])
        return res
```

### [78. 子集 (Subsets)](https://leetcode.cn/problems/subsets/description)

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(nums, tmp):
            res.append(tmp)
            if not nums:
                return
            for i in range(len(nums)):
                dfs(nums[i + 1:], tmp + [nums[i]])
        dfs(nums, [])
        return res
```

### [17. 电话号码的字母组合 (Letter Combinations of a Phone Number)](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description)

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        mp = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }

        res = []

        def dfs(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for c in mp[nums[0]]:
                dfs(nums[1:], tmp + c)

        dfs(digits, '')
        return res
```

### [39. 组合总和 (Combination Sum)](https://leetcode.cn/problems/combination-sum/description)

```python
class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        res = []
        def dfs(nums, tmp):
            if sum(tmp) == target:
                res.append(tmp)
                return
            for i in range(len(nums)):
                if nums[i] > target - sum(tmp):
                    break
                dfs(nums[i:], tmp + [nums[i]])
        dfs(nums, [])
        return res
```

### [22. 括号生成 (Generate Parentheses)](https://leetcode.cn/problems/generate-parentheses/description)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def dfs(remain_open, remain_close, tmp):
            if remain_open == 0 and remain_close == 0:
                res.append(tmp)
                return
            if remain_open > 0:
                dfs(remain_open - 1, remain_close + 1, tmp + '(')
            if remain_close > 0:
                dfs(remain_open, remain_close - 1, tmp + ')')
        dfs(n, 0, "")
        return res
```

### [79. 单词搜索 (Word Search)](https://leetcode.cn/problems/word-search/description)

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, index):
            if index == len(word):
                return True
            if not (
                0 <= i <= len(board) - 1
                and 0 <= j <= len(board[0]) - 1
            ):
                return False
            if board[i][j] != word[index]:
                return False
            tmp = board[i][j]
            board[i][j] = '#'
            flag = (
                dfs(i - 1, j, index + 1)
                or dfs(i + 1, j, index + 1)
                or dfs(i, j - 1, index + 1)
                or dfs(i, j + 1, index + 1)
            )
            board[i][j] = tmp
            return flag

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False
```

### [131. 分割回文串 (Palindrome Partitioning)](https://leetcode.cn/problems/palindrome-partitioning/description)

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def dfs(s, tmp):
            if not s:
                res.append(tmp)
                return
            for i in range(1, len(s) + 1):
                if s[:i] == s[:i][::-1]:
                    dfs(s[i:], tmp + [s[:i]])
        dfs(s, [])
        return res
```

### [51. N 皇后 (N-Queens)](https://leetcode.cn/problems/n-queens/description)

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        s = '.' * n
        def dfs(row, cols, poss, negs, tmp):
            if row == n:
                res.append(tmp)
                return
            for col in range(n):
                if (
                    col not in cols
                    and row + col not in poss
                    and row - col not in negs
                ):
                    dfs(
                        row + 1,
                        cols | {col},
                        poss | {row + col},
                        negs | {row - col},
                        tmp + [s[:col] + 'Q' + s[col + 1:]]
                    )
        dfs(0, set(), set(), set(), [])
        return res
```

## 二分查找

### [35. 搜索插入位置 (Search Insert Position)](https://leetcode.cn/problems/search-insert-position/description)

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left
```

### [74. 搜索二维矩阵 (Search a 2D Matrix)](https://leetcode.cn/problems/search-a-2d-matrix/description)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        i, j = 0, len(matrix[0]) - 1
        while i <= len(matrix) - 1 and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
        return False
```

### [34. 在排序数组中查找元素的第一个和最后一个位置 (Find First and Last Position of Element in Sorted Array)](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description)

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def search(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        start = search(nums, target)
        if start > len(nums) - 1 or nums[start] != target:
            return [-1, -1]
        end = search(nums, target + 1) - 1
        return [start, end]
```

### [33. 搜索旋转排序数组 (Search in Rotated Sorted Array)](https://leetcode.cn/problems/search-in-rotated-sorted-array/description)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def find_min(nums):
            left, right = 0, len(nums) - 2
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] < nums[-1]:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        def _search(nums, left, right):
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1
            if left <= len(nums) - 1 and nums[left] == target:
                return left
            else:
                return -1

        i = find_min(nums)
        if target <= nums[-1]:
            left, right = i, len(nums) - 1
        else:
            left, right = 0, i
        return _search(nums, left, right)
```

### [153. 寻找旋转排序数组中的最小值 (Find Minimum in Rotated Sorted Array)](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description)

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 2
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[-1]:
                right = mid - 1
            else:
                left = mid + 1
        return nums[left]
```

### [4. 寻找两个正序数组的中位数 (Median of Two Sorted Arrays)](https://leetcode.cn/problems/median-of-two-sorted-arrays/description)

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        for i in range(m + 1):
            j = (m + n + 1) // 2 - i
            left1_max = -inf if i == 0 else nums1[i - 1]
            left2_max = -inf if j == 0 else nums2[j - 1]
            left_max = max(left1_max, left2_max)

            right1_min = inf if i == m else nums1[i]
            right2_min = inf if j == n else nums2[j]
            right_min = min(right1_min, right2_min)

            if left_max <= right_min:
                if (m + n) % 2 == 0:
                    return (left_max + right_min) / 2
                else:
                    return left_max
        return 0
```

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

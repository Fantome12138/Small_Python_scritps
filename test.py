import os

'''
3. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。
'''
class Solution():
    def lengthOfLongestSubstring(self, s):
        if not s: return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len = 0
        cur_len = 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len:
                max_len = cur_len
            lookup.add(s[i])
        return max_len

s = Solution()
string = 'abcabcbbd'
print(s.lengthOfLongestSubstring(string))

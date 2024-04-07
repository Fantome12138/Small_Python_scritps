import re
import numpy as np

def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    return matrix[len(str1)][len(str2)]

def Levenshtein_Distance2(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1,len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i-1] == word2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
str1 = []
str2 = []
a = ['PSL646U', 'OAONOOOAO DO', '220V.', 'LOMOOUN', 'A小★W', 'PSL646U 线路保护测控装置', '信号压板', '过流段', 'oeanz/10105607.00', 'UI', '过流段', '时信导异常[对时信号常', '【时029/02/1010.0722', '过流段', '时信号异常（对时信号常', '零流一段', '零流段', '告', '课', '2', '?', '分位合位', 'GDNZ207626000141', 'PSL646U', '201801284722', '工作电源', 'S4C国电南自']
b = ['T645UX', 'GDNZ209700027812', '', '220V1A', 'A本田', 'Z009800003003', 'PST645UX变压器保护测控装置', '信号压板', '过流段', '?', '过流段', '2023/01/2408:27:19.00', '过流段', '2023/01', '08:27:19.127', '高侧零流', '羊常对', '言号异常]', '低侧零流', '过负荷', '压', '告', '电压', '速', '加', 'Q', '非电量1', '非电量2', '非电量3', '9', '非电量4', '复', '归', '分位合位']

from zhon.hanzi import punctuation
import string

str1 = ''.join(a)
# str1 = re.sub(r"[%s]+" %punc, "",str1)
# str1 = "".join(re.findall(r'\b\w+\b',str1))
# str1 = re.sub('\W*', '', str1)
for i in punctuation:
    str1 = str1.replace(i,'')
for c in string.punctuation:
    str1 = str1.replace(c,'')
str1 = re.sub('[a-zA-Z\d]','',str1)

str2 = ''.join(b)
for i in punctuation:
    str2 = str2.replace(i,'')
for c in string.punctuation:
    str2 = str2.replace(c,'')
str2 = re.sub('[a-zA-Z\d]','',str2)

print(str1, '\n', str2)
print(Levenshtein_Distance(str1, str2))
print(Levenshtein_Distance2(str1, str2))

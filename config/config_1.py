# -*- coding:utf-8 -*-
# @Time : 2022/7/24 11:01 下午
# @Author : Bin Bin Xue
# @File : config_1
# @Project : myutils

'''
    功能：实现将print输出到控制台的内容一并写入文件中
    参考链接：https://blog.csdn.net/u010158659/article/details/81671901
'''
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger('a.log', sys.stdout)
sys.stderr = Logger('a.log_file', sys.stderr)		# redirect std err, if necessary

# now it works
print( 'print something')

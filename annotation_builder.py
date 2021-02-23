from create import create
import sys
import argparse
import os
list=[]
def main():
    # picpath=input('please enter pictures direction as "C:\\folder" format:')
    picpath = "C:\\Users\\Larry\\PycharmProjects\\DVA485\\Image_dataset"
    # annpath = input('please enter annotation file path as "C:\\folder" format:')
    annpath = "C:\\Users\\Larry\\PycharmProjects\\DVA485\\annotation.txt"
    for file in os.listdir(picpath):
        list.append(os.path.join(picpath, file))
    for path in list:
        create(path,annpath)
if __name__=='__main__':
    main()
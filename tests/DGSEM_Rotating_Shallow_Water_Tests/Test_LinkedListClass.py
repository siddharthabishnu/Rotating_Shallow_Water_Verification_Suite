"""
Name: Test_LinkedListClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the linked list class defined in 
../../src/DGSEM_Rotating_Shallow_Water/LinkedListClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import LinkedListClass as LL


def TestLinkedList():
    myLinkedList = LL.LinkedList()
    e1 = LL.Record(listData=1,key=1)
    e2 = LL.Record(listData=2,key=2)
    e3 = LL.Record(listData=3,key=3)
    myLinkedList.head = e1
    myLinkedList.head.next = e2
    e2.next = e3
    myLinkedList.tail = e3
    e3.next = None
    print('Printing linked list before addition:')
    myLinkedList.PrintList()
    myLinkedList.AddToList(4,4)
    print('Printing linked list after addition:')
    myLinkedList.PrintList()
    print('Is the linked list empty?')
    LogicalOutput = myLinkedList.ListIsEmpty()
    print(LogicalOutput)
    print('Destructing linked list:')
    myLinkedList.DestructList()
    print('Is the linked list empty?')
    LogicalOutput = myLinkedList.ListIsEmpty()
    print(LogicalOutput)
    
    
do_TestLinkedList = False
if do_TestLinkedList:
    TestLinkedList()
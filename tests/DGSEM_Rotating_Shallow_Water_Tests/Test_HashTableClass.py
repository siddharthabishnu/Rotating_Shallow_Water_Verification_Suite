"""
Name: Test_HashTableClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the hash table class defined in ../../src/DGSEM_Rotating_Shallow_Water/HashTableClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import HashTableClass as HT


def TestHashTable():
    myHashTable = HT.HashTable(2)
    key1 = 0
    key2 = 1
    print('Is the linked list myHashTable.myTableData[0] empty?')
    print(myHashTable.myTableData[key1].ListIsEmpty())
    if not(myHashTable.ContainsKeys(key1,key2)):
        inData = 2
        myHashTable.AddDataForKeys(inData,key1,key2)
        print('Printing the linked list myHashTable.myTableData[0]:')
        myHashTable.myTableData[key1].PrintList()
    print('Does the hash table myHashTable contain the keys 0 and 1?')
    print(myHashTable.ContainsKeys(key1,key2))
    print('Is the linked list myHashTable.myTableData[0] empty?')
    print(myHashTable.myTableData[key1].ListIsEmpty())
    print('Destructing the linked list myHashTable.myTableData[0]:')
    myHashTable.DestructHashTable()
    print('Does the hash table myHashTable contain the keys 0 and 1?')
    print(myHashTable.ContainsKeys(key1,key2))
    print('Is the linked list myHashTable.myTableData[0] empty?')
    print(myHashTable.myTableData[key1].ListIsEmpty())
    

do_TestHashTable = False
if do_TestHashTable:
    TestHashTable()
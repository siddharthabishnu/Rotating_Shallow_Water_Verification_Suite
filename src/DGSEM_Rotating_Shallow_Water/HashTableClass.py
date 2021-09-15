"""
Name: HashTableClass.py
Author: Sid Bishnu
Details: This script contains functions for building a hash table class using an array of pointers to linked lists.
"""


import numpy as np
import sys
from IPython.utils import io
with io.capture_output() as captured:
    import LinkedListClass as LL


class HashTable:
    
    def __init__(myHashTable,N):
        myHashTable.myTableData = np.empty(N,dtype=LL.LinkedList)
        for i in range(0,N):
            myHashTable.myTableData[i] = LL.LinkedList()
        
    def DestructHashTable(myHashTable): # Takes in a list, rewinds it.
        N = len(myHashTable.myTableData)
        for i in range(0,N):
            myHashTable.myTableData[i].DestructList()
        
    def ContainsKeys(myHashTable,i,j):
        if myHashTable.myTableData[i].ListIsEmpty(): # This list has not been started.
            ItDoesContain = False
            return ItDoesContain
        # Rewind the list.
        myHashTable.myTableData[i].current = myHashTable.myTableData[i].head
        while myHashTable.myTableData[i].current is not None:
            thisData, thisKey = myHashTable.myTableData[i].GetCurrentData()
            if thisKey == j: # This list already has this key.
                ItDoesContain = True
                return ItDoesContain
            # Otherwise we move to the next element in the list.
            myHashTable.myTableData[i].MoveToNext()
        ItDoesContain = False
        return ItDoesContain
    
    def AddDataForKeys(myHashTable,inData,i,j):
        if myHashTable.myTableData[i].ListIsEmpty():
            # This table entry is not pointing to a linked list, so we construct this table entry.
            myHashTable.myTableData[i] = LL.LinkedList()
        if not(myHashTable.ContainsKeys(i,j)):
            # This table entry does not contain the keys (i,j), so we add to the list.
            myHashTable.myTableData[i].AddToList(inData,j)
            
    def GetDataForKeys(myHashTable,i,j):
        if myHashTable.myTableData[i].ListIsEmpty(): # This table entry is not pointing to a linked list.
            print('Script HashTableClass.py: Function GetDataForKeys:')
            print('Table entry is not associated for keys %d %d.' %(i,j))
            print('Stopping!')
            sys.exit()
        myHashTable.myTableData[i].current = myHashTable.myTableData[i].head
        while myHashTable.myTableData[i].current is not None:
            # This table entry does not contain the keys (i,j), so we add to the list.
            thisData, thisKey = myHashTable.myTableData[i].GetCurrentData()
            if thisKey == j:
                outData = thisData
                return outData
            myHashTable.myTableData[i].MoveToNext()
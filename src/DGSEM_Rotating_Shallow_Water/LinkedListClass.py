"""
Name: LinkedListClass.py
Author: Sid Bishnu
Details: This script contains functions for forming a linked list class with integer data.
"""


class Record:
    
    def __init__(myRecord,listData=0,key=0,PointerToNextRecord=None):
        myRecord.listData = listData
        myRecord.key = key
        myRecord.next = PointerToNextRecord


class LinkedList:
    
    def __init__(myLinkedList): # Takes in a list and points the head, tail, and current position to None.
        myLinkedList.head = None
        myLinkedList.tail = None
        myLinkedList.current = None   
        
    def DestructList(myLinkedList): # Takes in a list, rewinds it.
        myLinkedList.current = myLinkedList.head
        while myLinkedList.current is not None:
            pNext = myLinkedList.current.next # Point to the next in the list.
            # Set the pointer to the current position to None.
            myLinkedList.current = None
            # Update the current position.
            myLinkedList.current = pNext
        myLinkedList.head = myLinkedList.current
        myLinkedList.tail = myLinkedList.current   
        """
        In python garbage collection happens. Therefore, only 
        myLinkedList.head = None 
        myLinkedList.tail = None
        myLinkedList.current = None
        would also destruct the link list.
        """
    
    def ListIsEmpty(myLinkedList):
        LogicalOutput = (myLinkedList.head is None) and (myLinkedList.tail is None) and (myLinkedList.current is None)
        return LogicalOutput        
    
    def AddToList(myLinkedList,inData,inKey):
        newRecord = Record()
        if myLinkedList.tail is not None:
            myLinkedList.tail.next = newRecord
            myLinkedList.tail = newRecord
        else: # if the tail points to None
            myLinkedList.tail = newRecord
            myLinkedList.head = newRecord
        myLinkedList.current = myLinkedList.tail
        myLinkedList.current.next = None
        myLinkedList.current.listData = inData
        myLinkedList.current.key = inKey
        
    def GetCurrentData(myLinkedList):
        outData = myLinkedList.current.listData
        outKey = myLinkedList.current.key
        return outData, outKey
    
    def MoveToNext(myLinkedList):
        myLinkedList.current = myLinkedList.current.next
        
    def PrintList(myLinkedList):
        myLinkedList.current = myLinkedList.head
        while myLinkedList.current is not None:
            outData, outKey = myLinkedList.GetCurrentData()
            print(outData,outKey)
            myLinkedList.MoveToNext()
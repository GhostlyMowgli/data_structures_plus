#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:33:18 2018

@author: AHinkle
"""

# DataStructures_learned.py

class Node():
    def __init__(self, value=None, next_node = None):
        self.value = value
        self.next = next_node
    
    def get_value(self):
        return self.value
    
    def get_next(self):
        return self.next
    
    def set_next(self, new_next):
        self.next = new_next

class LinkedList:
    def __init__(self, head=None):
        self.head = head
    
    def insert(self, value):
        new_node = Node(value)
        new_node.set_next(self.head) # why can't I include self.head in the Node() function?
        self.head = new_node
    
    def append(self, value):
        current = self.head
        if current:
            while current.next:
                current = current.next
            current.next = Node(value)
        else:
            self.head = Node(value)
            # current = Node(value) # for some reason this doesn't work....!!! UGH
    
    def remove(self, value, start=0):
        current = self.head
        previous = self.head
        found = False
        if start > 0:
            for i in range(start):
                current = current.next
        while current:
            if current.value == value:
                if current == self.head:
                    self.head = current.next
                    # whether current.next is None or a value, we want to do this either way
                elif current.next:
                    previous.next = current.next
                else:
                    previous.next = None
                # print(str(value) + " has been successfully removed.")
                found = True
                break
            previous = current
            current = current.next
        if found == False:
            raise ValueError("value never found...")
    
    def pop(self):
        current = self.head
        if not current:
            raise ValueError("List is already empty.")
        elif not current.next:
            value = current.value
            current.value = None
            return value
        while current.next.next:
            current = current.next
        value = current.next.value
        current.next = None
        return value
    
    def partition(self,start=0,stop=-1):
        if self.size() < stop:
            if self.size() == stop - 1:
                stop -= 1
            else:
                raise ValueError("Linked List is smaller than the stop input. \
                                 The Linked List is " + str(self.size()) + " .\n\
                                 The stop specified is " + str(stop))
        elif stop == -1:
            stop = self.size()
            print("'stop' was left blank, so it was set to \
                  the size of the linked list.")
        elif stop == 0:
            return self
        
        # now that we know the inputs are good, continue
        ll_a = LinkedList()
        index = 0
        current = self.head
        while index < stop:
            if index >= start:
                ll_a.insert(current.value)
            index += 1
            current = current.next
        return ll_a
    
    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            # current = self.get_next()
            current = current.next
        return count
    
    def size_mine(self,current=None, count=0):
        """
        for a recursive approach to size
        I think this would use less memory than size()?
        """
        if count == 0:
            current = self.head
        if self.get_next():
            count += 1
            if current == self.head:
                current = self.head.get_next() # issue w/ inheritence
            else:
                current = self.get_next()
            return self.size_mine(current,count)
        else:
            return count + 1
    
    def show_all(self,current=''):
        if current == '':
            current = self.head
        if current:
            print(current.value)
            if current.next:
                return self.show_all(current.next)

class Stack():
    """
    This class is for creating a stack by leeraging
    a Node class to be the individual objects,
    and the stack class methods/functions just handle
    relationships between nodes and add/modify/delete them.
    """
    def __init__(self,value=None,next_node=None):
        self.top = Node(value,next_node)
        self.size = 0
    
    def add(self,value):
        n = Node(value,self.top)
        self.top = n
        self.size += 1
    
    def remove(self):
        self.size -= 1
        if self.top.next:
            n = self.top.value
            self.top = self.top.next
            return n
        else:
            n = self.top.value
            self.top = None
            return n
    
    def check_size(self):
        return self.size

class Queue():
    """
    This is the best working version I can come up with so far
    that is both my style and 100% working.
    
    I would ideally like the remove() method to be recursive,
    and I would like to fully understand the issues with my
    earlier attempts at Stacks/Queues.
    
    Another interesting approach with Queues specifically
    is to have variables for both the front and back of the queue.
    Instead of having to traverse fom one end to the other when
    adding or removing, you can quickly start at that end by having
    two class variables or instance variables of a queue w/ node
    objects.
    """
    def __init__(self,value=None,next_node=None):
        if value == None and next_node == None:
            self.tail = None # tail points to the last element to be removed
            self.size = 0
        else:
            self.tail = Node(value,next_node)
            self.size = 1
    
    def add(self,value):
        n = Node(value)
        n.next = self.tail # point the new node to link to the old tail
        self.tail = n # now have the tail point to the new node
        self.size += 1
    
    def remove(self, current=''):
        if current == '':
            current = self.tail
        if current.next:
            if current.next.next:
                current = current.next
                return self.remove(current) # recursive - keep going to front
            else:
                n = current.next.value
                current.next = None
                self.size -= 1
                return n
        elif current == self.tail:
            # now I'm wondering if this is even smart
            # shouldn't I be checking if current is a None type?!?!
            if current.value:
                n = current.value
                current = None
                self.size -= 1
                # print("Queue is now empty - returning final number")
                return n
            else:
                return "Queue is already empty."
        else:
            raise ValueError("mind boggling coding error...")
        
    def check_size(self):
        return self.size
    
    def show_all(self, current=''):
        if current == '':
            current = self.tail
        if current:
            if type(current) == int or type(current) == str:
                print(current.value)
            else:
                print(current.value.value)
            if current.next:
                return self.show_all(current.next)
        else:
            print("Empty queue.")


class TreeNode():
    def __init__(self,value):
        self.value = value
        self.left_child = None
        self.right_child = None

class MaxHeap2():
    def __init__(self,value=None):
        self.head = TreeNode(value)
    
    def insert(self,value,prev=None):
        new_node = TreeNode(value) # create the node
        cur = self.head
        if cur.left_child and not cur.right_child:
            cur.right_child = new_node
        elif cur.left_child and cur.right_child:
            prev = cur
            cur.insert(value,prev)
        else:
            pass

# Max Heap
class MaxHeap():
    def __init__(self,value):
        self.value = value
        self.left_child = None
        self.right_child = None
        self.head = self # trade off memory wise
    
    def insert_node(self,value):
        cur = self.head
        prev = self.head
        if cur:
            if value < cur.value:
                if cur.left_child and cur.right_child:
                    # look another level down?
                    pass
                # elif cur:
                    # pass

class BinaryTree():
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None
    
    def insert_left(self, value):
        if self.left_child == None:
            self.left_child = BinaryTree(value)
        else:
            new_node = BinaryTree()
            new_node.left_child = self.left_child
            self.left_child = new_node
    
    def insert_right(self, value):
        if self.right_child == None:
            self.right_child = BinaryTree(value)
        else:
            new_node = BinaryTree()
            new_node.right_child = self.right_child
            self.right_child = new_node
    
    def pre_order(self):
        print(self.value)
        
        if self.left_child:
            self.left_child.pre_order()
        
        if self.right_child:
            self.right_child.pre_order()
    
    def in_order(self):
        if self.left_child:
            self.left_child.in_order()
        print(self.value)
        if self.right_child:
            self.right_child.in_order()
            
    def post_oder(self):
        if self.left_child:
            self.post_order()
        if self.right_child:
            self.post_order()
        print(self.value)
    
    def bfs_mine(self):
        pass
    
    def dfs_mine(self):
        pass
    
    
    def bfs_online(self):
        # I'm not so sure about this bfs.
        # I should create my own from scratch
        queue = Queue()
        queue.add(self)
        
        while not queue.size > 0:
            
            current_node = queue.tail
            print(current_node.value)
            
            if self.left_child:
                queue.add(self.left_child)
            if self.right_child:
                queue.add(self.right_child)



class BinarySearchTree():
    
    def __init__(self,value):
        """
        we do NOT give the option to set the child
        nodes in the init, because we want to always
        use the insert_node rules to determine where
        each child node goes.
        """
        self.value = value
        self.left_child = None
        self.right_child = None
        self.count_children = 0
        
    def insert_node(self, value):
        self.count_children += 1
        if value <= self.value and self.left_child:
            self.left_child.insert_node(value)
        elif value <= self.value:
            self.left_child = BinarySearchTree(value)
        elif value >= self.value and self.right_child:
            self.right_child.insert_node(value)
        else:
            self.right_child = BinarySearchTree(value)
    
    def find_node(self, value):
        if value < self.value and self.left_child:
            return self.left_child.find_node(value)
        elif value > self.value and self.right_child:
            return self.right_child.find_node(value)
        else:
            return value == self.value
    
    def remove_node(self, value, parent=None):
        # Idk how the fuck I'm supposed to have a
        # parent node that doesn't have a default value
        # which is what was suggested online.
        # you can't possibly have a parent node on the root
        self.count_children -= 1
        if value < self.value and self.left_child:
            return self.left_child.remove_node(value, self)
        elif value < self.value:
            return False # value not in tree
        elif value > self.value and self.right_child:
            return self.right_child.remove_node(value, self)
        elif value > self.value:
            return False # value not in tree
        else:
            if self.left_child is None and self.right_child is None and self == parent.left_child:
                parent.left_child = None
                self.clear_node() # not sure this does anything, 
                # since we're removing the link to the parent node
                # the 'trash collection' in python will clear this out since no variable points to it
            elif self.left_child is None and self.right_child is None and self == parent.right_child:
                parent.right_child = None
                self.clear_node()
            elif self.left_child and self.right_child is None and self == parent.left_child:
                parent.left_child = self.left_child
                self.clear_node()
            elif self.left_child and self.right_child is None and self == parent.right_child:
                parent.right_child = self.left_child
                self.clear_node()
            elif self.left_child is None and self.right_child and self == parent.left_child:
                parent.left_child = self.right_child
                self.clear_node()
            elif self.left_child is None and self.right_child and self == parent.right_child:
                parent.right_child = self.right_child
                self.clear_node()
            else:
                # not yet clear why we take the lowest of the highs
                # rather than the highest of the lows
                self.value = self.right_child.find_minimum_value()
                self.right_child.remove_node(self.value, self)
            
            # self.count_children += 1 # undo the reduction
            # i need to apply this elsehwere in my bst
            return True
    
    
    def clear_node(self):
        self.value = None
        self.left_child = None
        self.right_child = None
    
    def find_minimum_value(self):
        if self.left_child:
            return self.left_child.find_minimum_value()
        else:
            return self.value
    
    def find_maximum_value(self):
        if self.right_child:
            return self.right_child.find_maximum_value()
        else:
            return self.value
    
    def dfs(self):
        if self.left_child:
            self.left_child.dfs()
        if self.right_child:
            self.right_child.dfs()
        print(self.value)
    
    def bfs(self):
        # still not YET working
        """
        This is currently VERY inefficient from a memory
        standpoint, as it adds a subtree for EACH node..
        """
        queue = Queue()
        queue.add(self)
        while queue.size > 0:
            current = queue.remove()
            print(current.value)

            if current.left_child:
                queue.add(current.left_child)
            if current.right_child:
                queue.add(current.right_child)
            
            # queue.remove won't print, because the object
            # in queue.tail is not an int but a bst object
            

class hashtable():
    def __init__(self, size=10):
        """
        size is the number of slots in the initial hash table.
        size should be roughly the number of elements we expect
        to add to this hash table / the number of unique results
        that will be returned from the hashfunc().
        """
        self.table = [[] for i in range(size)]
        self.size = size
    
    def hashfunc(self, k):
        if type(k) == str:
            x = len(k) % (self.size + 1)
        elif type(k) == int:
            x = k % (self.size + 1)
        else:
            raise ValueError("input needs to be a str or int")
        return x
        
    def add(self, k, v):
        """
        add or update key value pairs in the hashtable
        """
        x = self.hashfunc(k)
        self.table[x].append((k,v))
    
    def find(self, k):
        """
        find a specific key/value location in the hash table,
        and return that location.
        """
        x = self.hashfunc(k)
        for i in range(len(self.table[x])):
            if self.table[x][i][0] == k:
                # v = self.table[x][1] # valid but not needed here
                return (x,i)
            else:
                pass
        raise ValueError("key not found in hash table...")
    
    def remove(self, k):
        """
        remove a key and value completely from the hashtable
        """
        x, loc = self.find(k)
        r = self.table[x].pop(loc)
        return "key value pair (" + k + ", " + r + ") was successfully removed"
    
    def get_value(self, k):
        """
        return the value corresponding to a key, k.
        """
        x, y = self.find(k)
        return self.table[x][y][1]
    
    def show_all(self):
        for row in self.table:
            for item in row:
                print(item,)

########################################################
########################################################

# Algorithms on sorting and searching


# Insertion Sort
def InsertionSort(linkedlist, output=False):
    # working
    """
    for sorting a linked list by using a queue
    if output==True, this function returns the new LL;
    otherwise, it just modifies it locally.
    Big-O: N
    """
    # search and sort
    current = linkedlist.head
    ll_size = linkedlist.size() # static original linked list size
    queue = Queue() # create an empty queue
    
    while queue.check_size() < ll_size:
        current = linkedlist.head
        selected = linkedlist.head
        while current is not None:
            if current.value < selected.value:
                selected = current
            current = current.next
        queue.add(selected.value)
        linkedlist.remove(selected.value)
    
    # now rebuild the linked list
    while queue.check_size() > 0:
        linkedlist.append(queue.remove())
    
    if output == True:
        return linkedlist


# Bubble Sort
def BubbleSort(linked_list):
    # working
    """
    I need to think about the tradeoff of swapping
    with the highest recent value or the earliest value.
    
    Swapping with the earliest value has the advantage
    of reducing the number of passes each time.
    
    perhaps I create my own variation of BubbleSort
    that counts the distance between the selected
    and both the earliest and the highest and
    performs the swap that results in the smallest
    movement? or biggest?
    
    Depending how large the difference is between the
    numbers, it could have a beneficial or negative
    impact.
    
    Maybe I do a normal bubble sort on the first pass
    while also measuring the dispersion of values (max,
    min, average?), and takes into account the distance
    of each number from the max/min.
    """
    start = 0
    end = linked_list.size()
    while start < end:
        current = linked_list.head
        selected = linked_list.head
        previous = linked_list.head
        for i in range(start,end):
            if current.value < selected.value:
                # swap the values
                cv = current.value
                current.value = selected.value
                selected.value = cv
            elif current.value < previous.value:
                # this is my advent to make BubbleSort more efficient
                cv = current.value
                current.value = previous.value
                previous.value = cv
            else:
                pass
            previous = current
            current = current.next
        start += 1
    return linked_list

# Selection Sort
def SelectionSort(linked_list,inplace=True):
    # working
    """
    very simple sorting algorithm
    """
    if inplace == True:
        start = linked_list.head
        current = linked_list.head
        selected = linked_list.head
    else:
        new_list = linked_list
        start = new_list.head
        current = new_list.head
        selected = new_list.head
    while start:
        current = start
        selected = start
        while current:
            if current.value < selected.value:
                selected = current
            current = current.next
        if selected.value < start.value:
            s = start.value
            start.value = selected.value
            selected.value = s
        start = start.next
    return linked_list
    

# Quick Sort
def QuickSort(linked_list):
    # not yet working
    """
    One issue I may have is if the list has multiple
    nodes with the same value. If I have two "7's" and
    one is before the pivot and one is after, my
    ll.remove(value) function is going to find the first 7
    which is in front of the pivot and not the second 7
    which is after, and which is the one we want to remove
    
    I added a start to the ll.remove() function, so it
    starts looking for the value to remove after the
    index location indicated by 'start', which defaults
    to zero, so it starts at the beginning if you don't
    specify anything.
    
    Another issue is in how exactly QuickSort is supposed
    to work.. it seems the default is to insert moved
    values to just before the pivot or just after it.
    I'm not sure if it's a necessity or a preference to
    do it that way vs inserting everything at the
    far ends - very front and very back..
    """
    lls = linked_list.size()
    pivot = linked_list.head
    current = linked_list.head
    p = 0 # pivot index, so we know where to start
    # new_list = LinkedList()
    if lls > 1:
        while current:
            if current.value < pivot.value:
                cv = current.value
                linked_list.remove(cv,p)
                linked_list.insert(cv)
                p += 1
            current = current.next
            
        # now take the two halves and sort each
        lla = linked_list.partition(0,p)
        if lla.size() > 1:
            lla = QuickSort(lla)
        llb = linked_list.partition(p,lls)
        if llb.size() > 1:
            llb = QuickSort(llb)
        
        # previously I just had the two below
        # lla = QuickSort(linked_list.partition(0,p))
        # llb = QuickSort(linked_list.partition(p,lls))
        
        
        cb = llb.head
        while cb:
            lla.append(cb.value)
        return lla
    else:
        return linked_list
    
    # recursion issue w/ going infinitely deeper
    # maybe 


# Heap Sort
# related to Selection Sort
def HeapSort(linked_list):
    # not yet working
    """
    Step 1) create an empty binary tree
    Step 2) go through the LL and push everything to
        the Binary Tree.
    Step 3) rebuild the LL by extractinig each largest
        element from the binary tree.
    """
    current = linked_list.head
    heap = BinaryTree()
    while current:
        pass


# Merge Sort
def MergeSort(linked_list):
    # working
    """
    This MergeSort function takes a single linked list
    and creates sub linked lists to build
    a new, ordered, linked list.
    
    The issue I'm having now is that I first check if
    A is less than B and then insert A first before B or
    whatever else follows. But when I insert, it's putting
    the "trueest" (loweset) values towards the back,
    creating a reversed, sorted listed.
    So each time I traverse again, I'm working from two
    lists that were previously flipped sorted.
    
    I fixed the above issue by creating an 'append' fx
    for LinkedList that adds the value to the back of the 
    list which avoids the issue of flipping the sorted order
    back and forth.
    """
    lls = linked_list.size()
    if lls >= 2:
        lla = MergeSort(linked_list.partition(0,lls//2))
        llb = MergeSort(linked_list.partition(lls//2,lls))
    
        ca = lla.head # current node of linked list a
        cb = llb.head # current node of linked list b
    elif lls == 1:
        return linked_list
    else:
        lla = linked_list
        ca = lla.head
        cb = None
    
    new_list = LinkedList()
    while ca or cb:
        if ca and cb:
            if ca.value < cb.value:
                value = ca.value
                ca = ca.next
                lla.remove(value)
            else:
                value = cb.value
                cb = cb.next
                llb.remove(value)
        elif ca:
            value = ca.value
            ca = ca.next
            lla.remove(value)
        else:
            value = cb.value
            cb = cb.next
            llb.remove(value)
        new_list.append(value)
    return new_list



########################################################
########################################################


def random_list(size=10,minimum=0,maximum=100):
    """
    for quickly creating randomized linked lists
    so I can test sorting algos on them.
    """
    from random import randint
    linked_list = LinkedList()
    for i in range(size):
        linked_list.insert(randint(minimum,maximum))
    return linked_list

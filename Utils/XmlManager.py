# -*- coding: utf-8 -*-
'''
Created on 14/11/2014

@author: ucsp
'''

import elementtree.ElementTree as ET
from elementtree.ElementTree import ElementTree
from Preprocessing.commentPreprocessor import Comment_proccesor as CP

class XManager(object):
    
    def __init__(self , xml_file):
        self.__xml_file = xml_file
        self.__vector_comments = []
        self.__labels = []
        self.__data = []
    
    def set_file(self , xml_file):
        self.__xml_file = xml_file
    
    def get_data(self):
        tree = ET.parse(self.__xml_file)
        root = tree.getroot()
        for child in root:
            texto = child[1].text
            proc = CP(texto , True) 
            texto = proc.get_processed_comment()
            polaridad = child[2].text        
            self.__vector_comments.append(texto)
            self.__labels.append(polaridad)
        for i in range(len(self.__vector_comments)):
            value = (self.__vector_comments[i] , self.__labels[i])
            self.__data.append(value)                      
    
    def get_information(self , option):
        comments = []
        flags = []
        for i in range(len(self.__data)):
            val = self.__data[i][0]
            val2 = self.__data[i][1]
            if option == 0:                
                comments.append(val)
                flags.append(val2)
            elif option == 1:
                if val2!= 'NEU':
                    comments.append(val)
                    flags.append(val2)
            elif option == 2:
                if val2!= 'N':
                    comments.append(val)
                    flags.append(val2)
            elif option == 3:
                if val2!= 'P':
                    comments.append(val)
                    flags.append(val2)
        return [comments , flags]
        
        

if __name__ == '__main__':
    
    haber = XManager("peruvianTest.xml")
    haber.get_data() 
    inf = haber.get_information(0)
    print inf[0]
    print inf[1]
    

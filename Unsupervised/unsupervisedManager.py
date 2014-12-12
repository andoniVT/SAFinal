#!/usr/bin/env python
# -*- coding: utf-8 -*- 
'''
Created on 15/11/2014

@author: andoni
'''
import sys  
from sklearn.metrics.metrics import classification_report, confusion_matrix
reload(sys)  
sys.setdefaultencoding('utf8')

from Unsupervised.classifier import Classifier
from Configuration.settings import trainSpanish , trainPeruvian , testSpanish , testPeruvian
from Utils.XmlManager import XManager as XM

class UnSupervisedManager(object):
    
    def __init__(self , domain=""):
        self.__domain = domain
        self.__comments = []
        self.__labels = []
    
    def classify_comment(self, comment): 
        obj = Classifier()
        obj.classify(comment)
        sentiment = obj.get_label()
        return sentiment
    
    def classify_comments(self):
        pre = XM(self.__domain) 
        pre.get_data()
        testAll = pre.get_information(0)
        self.__comments = testAll[0] 
        self.__labels = testAll[1]  
                
        predicted = []
        obj = Classifier()
        for i in self.__comments:
            obj.classify(i)
            sentiment = obj.get_label()
            predicted.append(sentiment)
        return predicted
    
    def show_report(self, y_predicted):
        y_true = []
        y_predicted_new = []
        
        for i in range(len(self.__labels)):
            if self.__labels[i]=='P':
                y_true.append(1)
            if y_predicted[i]=='positivo':
                y_predicted_new.append(1)
            if self.__labels[i]=='N':
                y_true.append(-1)
            if y_predicted[i]=='negativo':
                y_predicted_new.append(-1)
            if self.__labels[i]=='NEU':
                y_true.append(0)
            if y_predicted[i]=='neutral':
                y_predicted_new.append(0)
        
        print classification_report(y_true, y_predicted_new)
        print confusion_matrix(y_true, y_predicted_new)


if __name__ == '__main__':
    
    manager = UnSupervisedManager(testSpanish)
    sentiments = manager.classify_comments()
    print sentiments
    manager.show_report(sentiments)
    
    
     

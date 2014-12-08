'''
Created on 08/12/2014

@author: andoni
'''

from Supervised.supervisedManager import SupervisedManager as SM
from sklearn.metrics import f1_score

def NaiveVoting(matrix):    
    positive = []
    negative = []
    neutral = []
    result = []    
    for i in range(len(matrix[0])):
        positive.append(0)
        negative.append(0)
        neutral.append(0)    
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 'P':
                positive[j]+=1
            elif matrix[i][j] == 'N':
                negative[j]+=1
            elif matrix[i][j] == 'NEU':
                neutral[j]+=1      
    for i in range(len(positive)):
        if (positive[i] > negative[i]) and (positive[i]>neutral[i]):
            result.append('P')
        elif (negative[i]>positive[i]) and (negative[i]>neutral[i]):
            result.append('N')
        else:
            result.append('NEU')
    return result

def max_value(labels_values):
    pos = 0
    max = 0
    for i in range(len(labels_values)):
        if max < labels_values[i][1]:
            max = labels_values[i][1]
            pos = i
    return labels_values[pos][0]
           
def weightedVoting():
    manager = SM()
    resultSVM = manager.optimize_classifierTFIDF(1,1)    
    resultNB = manager.optimize_classifierTFIDF(2, 1)
    resultME = manager.optimize_classifierTFIDF(3, 1)
    resultDT = manager.optimize_classifierTFIDF(4, 1)
    
    weightSVM = resultSVM[2]
    weightNB = resultNB[2]
    weightME = resultME[2]
    weightDT = resultDT[2]
    
    
    labelsSVM = resultSVM[0]
    labelsNB = resultSVM[0]
    labelsME = resultSVM[0]
    labelsDT = resultSVM[0]
    
    ytrue = resultSVM[1]
    
    
    labels_and_values = []
    for i in range(len(labelsSVM)):
        pos = ['P' , 0.0]
        neg = ['N' , 0.0]
        neu = ['NEU' , 0.0]
        vector = [pos,neg,neu]
        labels_and_values.append(vector)
        
    matrix_results = [labelsSVM, labelsNB, labelsME, labelsDT]
    weights = [weightSVM, weightNB, weightME, weightDT]
    
    for i in labels_and_values:
        print i
    
    for i in range(len(matrix_results)):
        for j in range(len(matrix_results[i])):
            if matrix_results[i][j] == 'P':
                labels_and_values[j][0][1] = labels_and_values[j][0][1] + weights[i]
            if matrix_results[i][j] == 'N':                
                labels_and_values[j][1][1] = labels_and_values[j][1][1] + weights[i]
            if matrix_results[i][j] == 'NEU':
                labels_and_values[j][2][1] = labels_and_values[j][2][1] + weights[i]
    
    finalResult = []
    for i in labels_and_values:
        label = max_value(i)
        finalResult.append(label)
    
    
    print "And the final table is: "
    manager.show_classificator_report(ytrue, finalResult)
    
    print "And the final results are:"
    print finalResult
    return finalResult
    
             
                
    
    
    

if __name__ == '__main__':
    
    weightedVoting()
   
    '''
    manager = SM()
    resultSVM = manager.optimize_classifierTFIDF(1, 1)
    
    print "este1"
    print resultSVM[0]
    print "este2"
    print resultSVM[1]
    print "este3"
    print resultSVM[2]
    #resultNB = manager.optimize_classifierTFIDF(2, 1)
    #resultME = manager.optimize_classifierTFIDF(3, 1)
    #resultDT = manager.optimize_classifierTFIDF(4, 1)
     
    print "Result SVM"
    print resultSVM[0] 
    print "Result NB"
    print resultNB[0]
    print "Result ME"
    print resultME[0]
    print "Result DT"
    print resultDT[0]
    
    
    matrix = [resultSVM[0] , resultNB[0] , resultME[0] , resultDT[0]]
    
    final_results = NaiveVoting(matrix)
    print "Final Result"
    print final_results
    
    y_true = resultSVM[1]
    
    manager.show_classificator_report(y_true, final_results)
    '''                                                                                                                                                        

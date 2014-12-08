'''
Created on 08/12/2014

@author: andoni
'''

''' path  '''
path_cicc = '/home/ucsp/workspace/SentimentAnalizerFinal'
path_cicc2 = '/home/ucsp/workspace'
path_home ='/home/andoni/Escritorio/PythonProjets/SentimentAnalizerFinal'
path_home2 = '/home/andoni/Escritorio/PythonProjets'

#path = path_cicc
path = path_home
path_classifiers = path_home2
#path_classifiers = path_cicc2

'''resource '''
big_text = path + '/Resource/big2.txt'
stop_words = path + '/Resource/stopwords_spanish.txt'

''' data test and training '''
trainSpanish = path + '/Data/train/spanishTrain.xml'
trainPeruvian = path + '/Data/train/peruvianTrain.xml'
testSpanish = path + '/Data/test/spanishTest.xml'
testPeruvian = path + '/Data/test/peruvianTest.xml'
prueba = path + '/Data/train/prueba.xml'

''' supervised classifiers '''
spanish_path = path_classifiers + '/DataTrained/classifierTrained/spanish'
spanishSVM = spanish_path + '/SVM'
spanishNaiveBayes = spanish_path + '/NaiveBayes'
spanishMaxEnt = spanish_path + '/MaxEnt'
spanishDecTree = spanish_path + '/DecTree'

peruvian_path = path_classifiers + '/DataTrained/classifierTrained/peruvian'
peruvianSVM = peruvian_path + '/SVM'
peruvianNaiveBayes = peruvian_path + '/NaiveBayes'
peruvianMaxEnt = peruvian_path + '/MaxEnt'
peruvianDecTree = peruvian_path + '/DecTree'
 
pclassAll = '/PNN_classifier.pk1'
pclassAllTF = '/PNN_TFclassifier.pk1'
pclassPNeg = '/PNeg_classifier.pk1'
pclassPNegTF = '/PNeg_TFclassifier.pk1'  
pclassPNeu = '/PNeu_classifier.pk1'
pclassPNeuTF = '/PNeu_TFclassifier.pk1'
pclassNegNeu = '/NegNeu_classifier.pk1'
pclassNegNeuTF = '/NegNeu_TFclassifier.pk1' 

 
spanishVectorModel = path_classifiers + '/DataTrained/vectorModel/spanish'
peruvianVectorModel = path_classifiers + '/DataTrained/vectorModel/peruvian'

pmodelAll = peruvianVectorModel + '/PosNegNeu'
pmodelPNeg = peruvianVectorModel + '/PosNeg'
pmodelPNeu = peruvianVectorModel + '/PosNeu'
pmodelNegNeu = peruvianVectorModel + '/NegNeu'

smodelAll = spanishVectorModel + '/PosNegNeu'
smodelPNeg = spanishVectorModel + '/PosNeg'
smodelPNeu = spanishVectorModel + '/PosNeu'
smodelNegNeu = spanishVectorModel + '/NegNeu'


nameCorpus = '/simpleCorpus.pk1'
nameTFCorpus = '/tfidfCorpus.pk1'
nameVectorizer = '/simpleVectorizer.pk1'
nameTFVectorizer = '/tfidfVectorizer.pk1'



''' unsupervised '''
# Recursos Lexicos 
#NEW_VOCABULARY = "path + '/resource/dictionary/groups/"
EMOTICONS = path + "/Resource/emoticons.txt"
BOOSTER_WORDS_SPANISH = path + "/Resource/booster_words_spanish.txt"
SENTIMENT_WORDS_SPANISH = path + "/Resource/sentiment_words_spanish.txt"
SLANGS_PERUVIAN = path + "/Resource/slangs_peruvian.txt"
NEGATING_WORDS_SPANISH = path + "/Resource/negating_words_spanish.txt"
COMBINATIONS_SPANISH  = path + "/Resource/combinations_spanish.txt"
COMBINATIONS_SLANGS_PERUVIAN = path + "/Resource/combinations_slangs_peruvian.txt"
PUNCTUATION = path + "/Resource/punctuation.txt"
STOPWORDS_SPANISH_OPINION_MINING = path + "/Resource/stopwords_spanish_opinion_mining.txt"

# Tipos de Terminos
TERM_TYPE_EMOTICON = 'emoticon'
TERM_TYPE_BOOSTER = 'booster'
TERM_TYPE_WORD_SLANG = 'word_slang'
TERM_TYPE_COMBINATION = 'combination'
TERM_TYPE_NEGATING = 'negating'
TERM_TYPE_PUNCTUATION = 'punctuation'
TERM_TYPE_NEUTRO = 'neutro'

# Simbolos adicionales
FLEXIS_SIMBOL = '#'
SPLITTER_WEIGHTS = '=='
ENCODING = 'utf-8'
TERM_NOT_FOUND = ''
SPLITTER_FREQUENT_WORD = '<##>'




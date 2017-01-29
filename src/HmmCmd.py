# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:22:58 2016

@author: Jose
"""

import os
import numpy as np
import sys

from Hmm import Hmm

class HmmCmd():
    """
    Class that represents manages the interaction with the user through the console as well as
    the I/O of files for the application
    
    Constants
    ----------
    """
    
    #Constants for the Operations
    GENSEQ_OP = 'genseq'
    VIT_OP = 'vit'
    TRAINHMM_OP = 'trainhmm'
    
    #Constants for the File Extensions
    TRANSITION_EXT = 'trans'
    EMISSION_EXT = 'emit'
    INPUT_EXT = 'input'
    
    def __init__(self, args):
        helpOperation = """ Hidden Markov Model Utilities:
        This program contains specific operations over HMMs, based on the Rabiner's HMM aproach.
        The following commands are available:
        genseq:     Generates  a collection of observation sequences with each sequence on a line.
                    It takes two parameters: 
                        - <name>    : the name of the HMM to work with
                        - <num_seq> : (optional) the number of observation sequences to generate. 
                                      Default: 10
                    e.g. genseq phone 12
                    This program uses the files in the folder <name>:
                        - <name>.trans : transition matrix structure
                        - <name>.emit  : emission matrix structure
                    As a result <num_seq> number of random observation sequences are shown based on the 
                    HMM structure.
        vit:        The vit operation finds the most probable sequences of states based on given 
                    observation sequences, as well as their probability, using the Viterbi algorithm.
                    It takes one parameter: 
                        - <name>    : the name of the HMM to work with
                    e.g vit phone
                    This program uses the files in the folder <name>:
                        - <name>.trans : transition matrix structure
                        - <name>.emit  : emission matrix structure
                        - <name>.input : file that contains the observation sequences
                    As a result the most probable sequence of states with its probability is shown 
                    for each sequence.
        trainhmm:   The trainhmm program trains the parameters of an HMM with a sequences of 
                    observations, using the Baum Welch algorithm.
                    It takes two parameters: 
                        - <name>    : the name of the HMM to work with
                        - <num_iter>: (optional) the maximum number iterations to run during 
                                      training. Default: 10
                    e.g. trainhmm 10
                    This program uses the files in the folder <name>:
                        - <name>.trans : transition matrix structure with the apriori probabilities
                        - <name>.emit  : emission matrix structure with the apriori probabilities
                        - <name>.input : file that contains the observation sequences to train the HMM
                    As a result the following files are created/overwritten, containing the structure
                    of the trained HMM:
                        - <name>_result.trans : the trained transition matrix structure
                        - <name>_result.emit  : the trained emission matrix structure
        --help:     Prints this help
                
        """
        
        #Parsing arguments (the first argument is always the name of the file)
        #in case no parameters are passed
        if len(args) <= 1:
            ###### FAST RUN FOR TEST!!
            folderPath = '..'
            hmmName ='phone'
            self.name, self.folder = hmmName, '..\\' + hmmName
            transMatrix, pi, statesDict = self.readTransitionMatrix()
            emissionMatrix, obsDict = self.readEmissionMatrix(statesDict)
            
            sequences = self.readInputSequences(obsDict)

            numIter = 10
            
            #Create the HMM
            hmm = Hmm(transMatrix, pi, emissionMatrix, statesDict, obsDict)
            #hmm.printHmm()
            
            #Calculate the state sequences based on the observations
            stateSequences = []
            for seq in sequences:
                viterbi = hmm.calculateViterbi(seq)
                stateSequences.append(viterbi)
            
            print ''
            self.printStatesSequences(sequences, stateSequences)
            
            
        
            #Train the HMM
            hmm.calculateBaumWelch(sequences, numIter)
            
            
            print 'New Pi'
            print hmm.pi
            print 'New trans'
            print hmm.transMatrix
            print 'New emi'
            print hmm.emissionMatrix
            
            
            transMatText = self.printTransitionMatrix(hmm)
            emMatText = self.printEmissionMatrix(hmm)
            
            # Prints on screen as well as in the files
            print ''
            print 'Trained Transitions:'
            print transMatText
            print 'Trained Emissions:'
            print emMatText
            
            
            self.printTextInFile(folderPath + os.sep + hmmName + '_result.' + self.TRANSITION_EXT, transMatText)
            self.printTextInFile(folderPath + os.sep + hmmName + '_result.' + self.EMISSION_EXT, emMatText)
            
            raise ValueError('FIN!!!')

            ##########
            raise ValueError('No arguments were passed to the application. Please use the command --help for more information.')
        
        #the first parameter is always the operation
        operation = args[1]
        
        #if exists the second parameter is always the HMM Name
        hmmName = ''
        if len(args) > 2:
            hmmName = args[2]
    
        #if exists the third parameter is always a number (num_seq or num_iter)
        numText = '10'
        if len(args) > 3:
            numText = args[3]
        
        if operation == '--help' or operation == 'help' or operation == 'h':
            print helpOperation
        else:
            #since the operation is not help then load the HMM structure
            
            #Check if the HMM name is empty
            if hmmName == '':
                raise ValueError('No HMM name was received.')
            
            #Find the correct path to the <name> folder
            folderPath = hmmName
            if(os.path.isdir(folderPath)):
                print 'Using HMM in: ' + folderPath
            else:
                folderPath =  os.getcwd() + os.sep + '..' + os.sep + folderPath
                if(os.path.isdir(folderPath)):
                    print 'Using HMM in: ' + folderPath
                else:
                    folderPath =  os.getcwd() + os.sep + folderPath
                    if(os.path.isdir(folderPath)):
                        print 'Using HMM in: ' + folderPath
                    else:
                        raise ValueError('HMM \'' + hmmName + '\' was not found.')
                
            #create the object HmmCmd and load the structure
            self.name, self.folder = hmmName, folderPath
            transMatrix, pi, statesDict = self.readTransitionMatrix()
            emissionMatrix, obsDict = self.readEmissionMatrix(statesDict)
            
            #proceed with the respective operation
            if operation == self.GENSEQ_OP:
                #Convert the number of sequences
                num_seq = 0
                try:
                    num_seq = int(numText)
                except ValueError:
                    raise ValueError('The value for num_seq \'' + numText + '\' is not valid.')
               
                #Create the HMM
                hmm = Hmm(transMatrix, pi, emissionMatrix, statesDict, obsDict)

                #Generate the observation sequences
                print ''
                seqGen = hmm.generateObservationSequences(num_seq)
                
                self.printObservationSequences(seqGen)
                
            elif operation == self.VIT_OP:
                
                #Read the observation sequences in the input file
                obsSequences = self.readInputSequences(obsDict)
                
                #Create the HMM
                hmm = Hmm(transMatrix, pi, emissionMatrix, statesDict, obsDict)
                
                #Calculate the state sequences based on the observations sequences
                stateSequences = []
                for obSeq in obsSequences:
                    viterbi = hmm.calculateViterbi(obSeq)
                    stateSequences.append(viterbi)
                
                print ''
                self.printStatesSequences(sequences, stateSequences)
                    
            elif operation == self.TRAINHMM_OP:
                #Convert the number of iterations
                numIter = 0
                try:
                    numIter = int(numText)
                except ValueError:
                    raise ValueError('The value for num_iter \'' + numText + '\' is not valid.')
               
                #Read the sequences in the input file
                sequences = self.readInputSequences(obsDict)
                
                #Create the HMM
                hmm = Hmm(transMatrix, pi, emissionMatrix, statesDict, obsDict)
            
                #Train the HMM
                hmm.calculateBaumWelch(sequences, numIter)
                
                transMatText = self.printTransitionMatrix(hmm)
                emMatText = self.printEmissionMatrix(hmm)
                
                # Prints on screen as well as in the files
                print ''
                print 'Trained Transitions:'
                print transMatText
                print 'Trained Emissions:'
                print emMatText
                
                self.printTextInFile(folderPath + os.sep + hmmName + '_result.' + self.TRANSITION_EXT, transMatText)
                self.printTextInFile(folderPath + os.sep + hmmName + '_result.' + self.EMISSION_EXT, emMatText)
                
            else:
                raise ValueError('No valid operation was received. Use --help for help. Operation received: \'' + operation + '\'')
            
    
    def readTransitionMatrix(self, statesDict = {}):
        """Reads the configuration file for the transition matrix. If no states dictionary is passed the it creates it from it."""
        transFile = self.folder + os.sep + self.name + '.' + self.TRANSITION_EXT
        
        transMatrix = np.matrix(0)
        
        with open(transFile) as f:
            lines = f.readlines()
            
            for line in lines:
                # Clean the lines and only work if is not empty
                line = line.strip()
                if line != '':
                    lineVals = line.split('\t')
                    
                    if(len(lineVals) == 1): #Handle the case of the first line (this can be skipped in the file)
                        #Add the name of the state to the dictionary   
                        if lineVals[0] not in statesDict:
                            statesDict[lineVals[0]] = len(statesDict)
                    elif(len(lineVals) == 3): #Handle the normal case
                        addValue = False
                        
                        #Add the name of the state (pos 0) to the dictionary if not already there
                        if lineVals[0] not in statesDict:
                            statesDict[lineVals[0]] = len(statesDict)
                            addValue = True
                        #Add the name of the state (pos 1) to the dictionary if not already there
                        if lineVals[1] not in statesDict:
                            statesDict[lineVals[1]] = len(statesDict)
                            addValue = True
                        # If a new value was added to the dictionary, add a new row and column to the matrix. 
                        if addValue:
                            transMatrix = np.append(transMatrix, np.zeros((len(statesDict) - 1, 1)), axis=1)
                            transMatrix = np.append(transMatrix, np.zeros((1, len(statesDict))), axis=0)
                        
                        #Assignt the transition value.
                        transMatrix[statesDict[lineVals[0]], statesDict[lineVals[1]]] = lineVals[2]
                    
                    else:
                        raise ValueError('Error reading transition matrix. Line \'' + line + '\' not understood.')
                        
        # get the Pi vector
        pi = np.squeeze(np.asarray( transMatrix[statesDict['INIT'], ]))
        
        # Remove the INIT row and columns
        pi = pi[1:]
        transMatrix = transMatrix[1:, 1:]
        del statesDict[Hmm.INIT_STATE]
        for key in statesDict.keys():
            statesDict[key] = statesDict[key] - 1
        

        #Return both the transition matrix and the state dictionary        
        return transMatrix, pi, statesDict
                       
    def readEmissionMatrix(self, statesDict = {}):
        """Reads the configuration file for the transition matrix. If no states dictionary is passed the it creates it from it."""
        transFile = self.folder + os.sep + self.name + '.' + self.EMISSION_EXT
        
        obsDict = {}
        emissionMatrix = np.zeros((len(statesDict), 0))
        
        
        with open(transFile) as f:
            lines = f.readlines()
            
            for line in lines:
                # Clean the lines and only work if is not empty
                line = line.strip()
                if line != '':
                    lineVals = line.split('\t')
                    
                    if(len(lineVals) == 3): #Handle the normal case
                        
                        #Add the name of the state (pos 0) to the dictionary if not already there
                        if lineVals[0] not in statesDict:
                            statesDict[lineVals[0]] = len(statesDict)
                            emissionMatrix = np.append(emissionMatrix, np.zeros((1, len(obsDict))), axis=0)
                            
                        #Add the name of the state (pos 1) to the dictionary if not already there
                        if lineVals[1] not in obsDict:
                            obsDict[lineVals[1]] = len(obsDict)
                            emissionMatrix = np.append(emissionMatrix, np.zeros((len(statesDict), 1)), axis=1)
                            
                        
                        #Assignt the transition value.
                        emissionMatrix[statesDict[lineVals[0]], obsDict[lineVals[1]]] = lineVals[2]
        
        #Return both the transition matrix and the state dictionary        
        return emissionMatrix, obsDict
                    
    def readInputSequences(self, obsDict = {}):
        """Reads the file with the input sequences for the HMM"""
        inputFile = self.folder + os.sep + self.name + '.' + self.INPUT_EXT
        sequences = []
        with open(inputFile) as f:
            lines = f.readlines()
            for line in lines:
                # Clean the lines and only work if is not empty
                line = line.strip()
                if line != '':
                    lineVals = line.split()
                    # Check that every observation is in the observation dictionary
                    for ob in lineVals:
                        if ob not in obsDict:
                            raise ValueError('Observation \'' + ob + '\' (in file \'' + self.name 
                                              + '.' + self.INPUT_EXT + '\') was not defined in the emmision file.')
                    
                    #Append the current observations to the sequences list
                    sequences.append(lineVals)
        
        return sequences
        
    def printObservationSequences(self, obsSequences):
        for seq in obsSequences:
            print ' '.join(seq)
            
    def printStatesSequences(self, obsSequences, stateSequences):
        
        for i in range(len(obsSequences)):
            obsSeq = obsSequences[i]
            stateSeq = stateSequences[i][1]
            prob = stateSequences[i][2]
            
            print 'P(path)={0:.5f}'.format(prob)
            print 'path:'
            
            for j in range(len(obsSeq)):
                print obsSeq[j] + '\t' + stateSeq[j]

    def printTransitionMatrix(self, hmm):
        
        transMatText = hmm.INIT_STATE + '\n'
        orderedStates = hmm.getOrderedDictionaryKeysByValue(hmm.statesDict)
        
        #Prints all the cases for the pi vector
        for i in range(len(hmm.pi)):
            if hmm.pi[i] > 0:
                transMatText += hmm.INIT_STATE + '\t' + orderedStates[i] + '\t' + self.getFloatAsText(hmm.pi[i]) + '\n'

        #Prints the rest values from the matrix itself
        (rows,cols) = hmm.transMatrix.shape
        for i in range(rows):
            for j in range (cols):
                if hmm.transMatrix[i,j] > 0:
                    transMatText += orderedStates[i] + '\t' + orderedStates[j] + '\t' + self.getFloatAsText(hmm.transMatrix[i,j]) + '\n'
                
        return transMatText
        
    def printEmissionMatrix(self, hmm):
        
        emMatText = ''
        orderedStates = hmm.getOrderedDictionaryKeysByValue(hmm.statesDict)
        orderedEmissions = hmm.getOrderedDictionaryKeysByValue(hmm.emissionsDict)
        
        (rows,cols) = hmm.emissionMatrix.shape
        for i in range(rows):
            for j in range (cols):
                if hmm.emissionMatrix[i,j] > 0:
                    emMatText += orderedStates[i] + '\t' + orderedEmissions[j] + '\t' + self.getFloatAsText(hmm.emissionMatrix[i,j]) + '\n'
        
        return emMatText
    
    def getFloatAsText(self, floatNum):
        return '{0:.6f}'.format(floatNum).rstrip('0').rstrip('.')
    
    def printTextInFile(self, filePath, text):
        file = open(filePath, "w")
        file.write(text)
        file.close()

if __name__ == '__main__':
    args = sys.argv
    HmmCmd(args)
   
    
    
    

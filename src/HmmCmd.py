# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:22:58 2016

@author: Jose Millan
"""

import os
import sys

from Hmm import Hmm
from HmmIO import HmmIO

class HmmCmd():
    """
    Class that represents manages the interaction with the user through the console as well as
    the I/O of files for the application
    
    Attributes
    ----------
    hmmName : string
        The name of the current HMM model (the folder's name)
    folderPath : string
        The path to the HMM model's folder
    
    Constants
    ----------
    GENSEQ_OP : string
        Operation for the Generate Sequences Functionality
    VIT_OP : string
        Operation for the Viterbi Functionality
    TRAINHMM_OP : string
        Operation for the Train HMM with Baum-Welch Functionality
    """
    
    #Constants for the Operations
    GENSEQ_OP = 'genseq'
    VIT_OP = 'vit'
    TRAINHMM_OP = 'trainhmm'
    
    
    def __init__(self, args):
        """
        Constructor of the HMMCmd class. This function manages the input from the user in the console
        and launches the respective operations of the HMM program.
        
        Parameters
        ----------
        args : array of strings
            Arguments of the application. See the help text for more information.
        """
        
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
            self.hmmIO = HmmIO(hmmName)
            self.hmmName, folderPath = hmmName, self.hmmIO.folderPath
            transMatrix, pi, statesDict = self.hmmIO.readTransitionMatrix()
            emissionMatrix, obsDict = self.hmmIO.readEmissionMatrix(statesDict)
            
            sequences = self.hmmIO.readInputSequences(obsDict)

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
            
            
            self.hmmIO.writeTextInFile(folderPath + os.sep + hmmName + '_result.' + self.TRANSITION_EXT, transMatText)
            self.hmmIO.writeTextInFile(folderPath + os.sep + hmmName + '_result.' + self.EMISSION_EXT, emMatText)
            
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
                
            #Assign the hmmName to the class attribute
            self.hmmName = hmmName
            
            #Create the HmmIO object
            self.hmmIO = HmmIO(hmmName)
            
            #Find the correct path to the HMM folder
            folderPath = self.hmmIO.folderPath
            
            #Read the HMM objects from the files
            transMatrix, pi, statesDict = self.hmmIO.readTransitionMatrix()
            emissionMatrix, obsDict = self.hmmIO.readEmissionMatrix(statesDict)
            
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
                obsSequences = self.hmmIO.readInputSequences(obsDict)
                
                #Create the HMM
                hmm = Hmm(transMatrix, pi, emissionMatrix, statesDict, obsDict)
                
                #Calculate the state sequences based on the observations sequences
                stateSequences = []
                for obSeq in obsSequences:
                    viterbi = hmm.calculateViterbi(obSeq)
                    stateSequences.append(viterbi)
                
                print ''
                self.printStatesSequences(obsSequences, stateSequences)
                    
            elif operation == self.TRAINHMM_OP:
                #Convert the number of iterations
                numIter = 0
                try:
                    numIter = int(numText)
                except ValueError:
                    raise ValueError('The value for num_iter \'' + numText + '\' is not valid.')
               
                #Read the sequences in the input file
                sequences = self.hmmIO.readInputSequences(obsDict)
                
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
                
                self.hmmIO.writeTextInFile(folderPath + os.sep + hmmName + '_result.' + HmmIO.TRANSITION_EXT, transMatText)
                self.hmmIO.writeTextInFile(folderPath + os.sep + hmmName + '_result.' + HmmIO.EMISSION_EXT, emMatText)
                
            else:
                raise ValueError('No valid operation was received. Use --help for help. Operation received: \'' + operation + '\'')
            
        
    def printObservationSequences(self, obsSequences):
        """
        Function that prints in the console output the formated observation sequences
        
        Parameters
        ----------
        obsSequences : array of strings
            The array of observation sequences to be printed
        """
        for seq in obsSequences:
            print ' '.join(seq)
            
    def printStatesSequences(self, obsSequences, stateSequences):
        """
        Function that prints in the console output the formated states with their respective observations
        
        Parameters
        ----------
        obsSequences : array of strings
            The array of observation sequences to be printed
        stateSequences : array of strings
            The array of states sequences to be printed
        """
        for i in range(len(obsSequences)):
            obsSeq = obsSequences[i]
            stateSeq = stateSequences[i][1]
            prob = stateSequences[i][2]
            
            print 'P(path)={0:.5f}'.format(prob)
            print 'path:'
            
            for j in range(len(obsSeq)):
                print obsSeq[j] + '\t' + stateSeq[j]


    def printTransitionMatrix(self, hmm):
        """
        Function that prints in the console output the formated trasition matrix of an HMM model
        
        Parameters
        ----------
        hmm : Hmm object
            The Hmm model from where the transition matrix is going to be formatted and printed
        """
        
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
        """
        Function that prints in the console output the formated emission matrix of an HMM model
        
        Parameters
        ----------
        hmm : Hmm object
            The Hmm model from where the emission matrix is going to be formatted and printed
        """
        
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
        """
        Function that formats a float number so it contains maximum 6 decimals. It removes ending 
        zeros and if the number has no decimals it types its integer representation (no dot at the end)
        
        Parameters
        ----------
        floatNum : float
            The number to be formatted
        
        Returns
        -------
        floatText : string
            The string representation of the number with maximum 6 decimals and no ending 0 or dot.
        """
        return '{0:.6f}'.format(floatNum).rstrip('0').rstrip('.')
    


if __name__ == '__main__':
    """Main function of the HMMCmd. It creates a HmmCmd and runs it with the system's arguments"""
    args = sys.argv
    HmmCmd(args)
   
    
    
    

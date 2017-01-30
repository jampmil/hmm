# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:22:58 2016

@author: Jose Millan
"""

import os
import numpy as np
from Hmm import Hmm

class HmmIO():
    """
    Class that represents manages the I/O of files for the application
    
    Attributes
    ----------
    hmmName : string
        The name of the current HMM model (the folder's name)
    folderPath : string
        The path to the HMM model's folder
    
    Constants
    ----------
    TRANSITION_EXT : string
        Extension for the transition probabilities file
    EMISSION_EXT : string
        Extension for the emission probabilities file
    INPUT_EXT : string
        Extension for the input file
    """
    
    #Constants for the File Extensions
    TRANSITION_EXT = 'trans'
    EMISSION_EXT = 'emit'
    INPUT_EXT = 'input'
    
    
    
    def __init__(self, hmmName):
        """
        Constructor of the HmmIO class. When calling this function the program will
        search for the model's folder and assign it in the attribute folderPath.
        
        Parameters
        ----------
        args : array of strings
            Arguments of the application. See the help text for more information.
        """
        self.hmmName = hmmName
        self.folderPath = self.findModelFolder()

        
    def findModelFolder(self):
        """
        Function that finds the correct path to the hmm folder.
        
        Returns
        -------
        folderPath : string
            Path to the HMM folder
        """
        
        folderPath = self.hmmName
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
                    raise ValueError('HMM \'' + self.hmmName + '\' was not found.')
            
        #returns the folder path
        return folderPath
    
    
    def readTransitionMatrix(self, statesDict = {}):
        """
        Reads the configuration file for the transition matrix. If no states dictionary 
        is passed the it creates it from it.
        
        Parameters
        ----------
        statesDict : dictionary <string, int>
            Dictionary that links the name of the state with its position (pointer). If empty is created.

        Returns
        -------
        transMatrix : Numpy Matrix
            Transision matrix of the model
        pi : float array 
            Array of initial probabilities
        statesDict : dictionary {string:int}
            Dictionary that links the name of the state with its position (pointer)
        """
        transFile = self.folderPath + os.sep + self.hmmName + '.' + self.TRANSITION_EXT
        
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
        """
        Reads the configuration file for the transition matrix. If no states dictionary is passed the
        it creates it from the elements in the file.
        
        Parameters
        ----------
        statesDict : dictionary <string, int>
            Dictionary that links the name of the state with its position (pointer). If empty is created.

        Returns
        -------
        emissionMatrix : Numpy Matrix
            Emission matrix of the model
        emissionsDict : dictionary {string:int}
            Dictionary that links the name of the emission/observation with its position (pointer)        
        """
        transFile = self.folderPath + os.sep + self.hmmName + '.' + self.EMISSION_EXT
        
        emissionsDict = {}
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
                            emissionMatrix = np.append(emissionMatrix, np.zeros((1, len(emissionsDict))), axis=0)
                            
                        #Add the name of the state (pos 1) to the dictionary if not already there
                        if lineVals[1] not in emissionsDict:
                            emissionsDict[lineVals[1]] = len(emissionsDict)
                            emissionMatrix = np.append(emissionMatrix, np.zeros((len(statesDict), 1)), axis=1)
                            
                        
                        #Assignt the transition value.
                        emissionMatrix[statesDict[lineVals[0]], emissionsDict[lineVals[1]]] = lineVals[2]
        
        #Return both the transition matrix and the state dictionary        
        return emissionMatrix, emissionsDict
                    
    def readInputSequences(self, emissionsDict = {}):
        """
        Reads the file with the input sequences for the HMM
        
        Parameters
        ----------
        emissionsDict : dictionary <string, int>
            Dictionary that links the name of the emissions with its index (pointer).

        Returns
        -------
        sequences : array of string arrays
            The sequences of observations present in the input file
        """
        inputFile = self.folderPath + os.sep + self.hmmName + '.' + self.INPUT_EXT
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
                        if ob not in emissionsDict:
                            raise ValueError('Observation \'' + ob + '\' (in file \'' + self.hmmName 
                                              + '.' + self.INPUT_EXT + '\') was not defined in the emmision file.')
                    
                    #Append the current observations to the sequences list
                    sequences.append(lineVals)
        
        return sequences
    
        
    def writeTextInFile(self, filePath, text):
        """
        Writes a given text in a file.
        
        Parameters
        ----------
        filePath : string
            The path to the file to write
        text : string
            The text to write

        """
        file = open(filePath, "w")
        file.write(text)
        file.close()


    
    
    

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:01:40 2016

@author: Jose Millan
"""

import numpy as np

class Hmm():
    """
    Class that represents a Hidden Markov Model
    
    Attributes
    ----------
    transMatrix : Numpy Matrix
        Transision matrix of the model
    pi : float array 
        Array of initial probabilities
    emissionMatrix : Numpy Matrix
        Emission matrix of the model
    statesDict : dictionary {string:int}
        Dictionary that links the name of the state with its position (pointer)
    emissionsDict : dictionary {string:int}
        Dictionary that links the name of the emission/observation with its position (pointer)
    
    Constants
    ----------
    INIT_STATE : string
        Name of the default initial state of the model
    FINAL_STATE : string
        Name of the default final state of the model
    """
    
    INIT_STATE = 'INIT'
    FINAL_STATE = 'FINAL'
    
    def __init__(self, transMatrix, pi, emissionMatrix, statesDict = {}, emissionsDict = {}):
        """
        Constructor of the HMM class
        
        Parameters
        ----------
        transMatrix : Numpy Matrix
            Transision matrix of the model
        pi : float array 
            Array of initial probabilities
        emissionMatrix : Numpy Matrix
            Emission matrix of the model
        statesDict : dictionary {string:int}
            Dictionary that links the name of the state with its position (pointer)
        emissionsDict : dictionary {string:int}
            Dictionary that links the name of the emission/observation with its position (pointer)

        """
        self.transMatrix = transMatrix
        self.pi = pi
        self.emissionMatrix = emissionMatrix
        self.statesDict = statesDict
        self.emissionsDict = emissionsDict
        
        
    def printHmm(self):
        """Prints the HMM objects in the console. Used for debugging"""
        
        print "Transition Matrix:"
        print self.transMatrix
        print "Pi Vector:"
        print self.pi
        print "States Dictionary:"
        print self.statesDict
        print "Emission Matrix:"
        print self.emissionMatrix
        print "Emissions Dictionary:"
        print self.emissionsDict

    
    def generateObservationSequences(self, num):
        """
        Generate observation sequences based on the HMM attributes
        
        Parameters
        ----------
        num : int
            Number of sequences to generate
        
        Returns
        -------
        obsSequences : array of int
            The observation sequences generated
        """
        
        #Vector to store the sequences
        obsSequences = []

        for i in range(num):
            #Current Sequence and emissions Pointers
            seqPointers = []
            emiPointers = []
            
            #end state pointer
            endStatePointer = self.statesDict[self.FINAL_STATE]
            
            #get random state from the Pi vector and appended to the sequence
            nextState = self.drawFrom(self.pi)
            seqPointers.append(nextState)
    
            #loop until it reaches the END state
            stop = False
            while not stop: 
                if nextState == endStatePointer:
                    stop = True
                else:
                    #get random emission from the state
                    currentEmission = self.drawFrom(self.unstackMatrix(self.emissionMatrix[nextState, ]))
                    emiPointers.append(currentEmission)
                    
                    #get next state pointer
                    nextState = self.drawFrom(self.unstackMatrix(self.transMatrix[nextState, ]))
                    seqPointers.append(nextState)
                    
            #Append the found sequence to the final sequences
            obsSequences.append(self.translateEmissionPointers(emiPointers))
        
        #Return the observation sequences generated
        return obsSequences
             
    
    def calculateForwardBackward(self, obsSequence, A, B, pi):
        """
        Calculates the alpha and beta matrices using the forward-backward algorithm
        
        Parameters
        ----------
        obsSequence : array of string
            Observation sequence with the names of the emissions
        A : Numpy Matrix
            Transision matrix of the model
        B : Numpy Matrix
            Emission matrix of the model
        pi : float array 
            Array of initial probabilities
        
        Returns
        -------
        alpha : Numpy matrix
            The alpha matrix
        beta : Numpy matrix
            The beta matrix
        logProbability : float
            The logarithmic probability of the sequence
        """
        
        
        #Number of states
        numStates = B.shape[0]
        
        #Number of emissions in the current observation
        numObs = len(obsSequence)
        
        #array with the probabilities
        c = np.zeros([numObs])
        
        #Create alpha and beta matrices
        alpha = np.zeros([numStates, numObs])
        beta = np.zeros([numStates, numObs])
        
        #Array with the pointers of the emissions
        emissionPointers = self.translateEmissionToPointers(obsSequence)
        
        ### Forward part
        
        #Get the first values of alpha from the Pi vector
        alpha[:, 0] = pi * B[:, emissionPointers[0]]
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= c[0]
        
        #iterate over the emissions of the observation
        for i in np.arange(1, numObs):
            alpha[:, i] = np.dot(alpha[:, i - 1], A) * B[:, emissionPointers[i]]
            c[i] = 1.0 / np.sum(alpha[:, i])
            alpha[:, i] *= c[i]

        ### Backward part
        beta[:, numObs - 1] = c[numObs - 1]
        for i in np.arange(numObs - 1, 0, -1):
            beta[:, i - 1] = np.dot(A, B[:, emissionPointers[i]] * beta[:, i])
            beta[:, i - 1] *= c[i - 1]

        #calculate the final probability (log)
        logProbability = -(np.sum(np.log(c)))
            
        # return the alpha, beta matrices along with the log probabilities
        return alpha, beta, logProbability
    
    
    def calculateBaumWelch(self, obsSequences, maxIter = 10):
        """
        Trains the parameters of the HMM with the sequences of observations, using the Baum Welch algorithm.
        The function updates the attributes transMatrix, pi, emissionMatrix of the HMM.
        
        Parameters
        ----------
        obsSequences : array of string arrays
            Observation sequences to train the HMM
        maxIter : int
            Maximum number of iterations to perform
        """
        
        #Print the header of the training
        print '\nTraining with Baum-Welch up to ' + str(maxIter) + ' iterations.'
        
        #Convert the HMM matrices to numpy arrays for the algorithm
        A = np.asarray(self.transMatrix)        
        B = np.asarray(self.emissionMatrix)
        
        #Initial probabilities vector
        pi = self.pi

        #Get the variables for the algorithm
        numStates = self.emissionMatrix.shape[0]
        numEmissions = self.emissionMatrix.shape[1]
        
        #Get an array with the text value of the Emissions
        orderedEmissions = self.getOrderedDictionaryKeysByValue(self.emissionsDict)

        #Array for storing the probabilities
        probabilities = []
        
        #Iterate the until the max number of iterations is reached or until it's not improving
        for iteration in np.arange(maxIter):
            #Current probability
            currentProbability = 0
            
            #Create the elements to calculate the new A and B (separated in numerator and denominator)
            newA_num = np.zeros([numStates, numStates])
            newA_den = np.zeros([numStates])
            newB_num = np.zeros([numStates, numEmissions])
            newB_den = np.zeros([numStates])
            newPi = np.zeros([numStates])

            for obsSeq in obsSequences:

                #Calculate alpha and beta
                alpha, beta, logProbability = self.calculateForwardBackward(obsSeq, A, B, pi)
                currentProbability += logProbability
                
                #Number of emissions in the current observation
                numObs = len(obsSeq)
                
                #Array with the pointers of the emissions
                emissionPointers = self.translateEmissionToPointers(obsSeq)

                #Calculate gamma and normalize it
                gamma_raw = alpha * beta
                gamma = gamma_raw / gamma_raw.sum(0)

                #Calculate Xi
                xi = np.zeros([numStates, numStates, numObs - 1])
                for j in np.arange(numObs - 1):
                    for i in np.arange(numStates):
                        xi[i, :, j] = alpha[i, j] * A[i, :] * \
                            B[:, emissionPointers[j + 1]] * beta[:, j + 1]

                B_bar = np.zeros([numStates, numEmissions])
                for k in np.arange(numEmissions):
                    indicator = np.array([orderedEmissions[k] == x for x in obsSeq])
                    B_bar[:, k] = gamma.T[indicator, :].sum(0)

                #update the new elements using gamma and xi
                newA_num += xi.sum(2)
                newA_den += gamma.sum(1)
                newB_num += B_bar
                newB_den += gamma.sum(1)
                newPi += gamma[:, 0]                
           
            #Updates the A matrix
            A_bar = np.zeros([numStates, numStates])
            for i in np.arange(0, numStates - 1):
                A_bar[i, :] = newA_num[i, :] / newA_den[i]
            A = A_bar
           
            #Updates the B matrix
            for i in np.arange(0, numStates - 1):
                if newB_den[i] > 0:
                    newB_num[i, :] = newB_num[i, :] / newB_den[i]
                else:
                    newB_num[i, :] = newB_num[i, :]
            B = newB_num
            
            #Updates Pi vector
            pi[:] = newPi / np.sum(newPi)
            
            #TODO: Print the current probability
            textIteration = 'Iteration ' + str(iteration + 1) + ' totalLogProb=' 
            textIteration += ('{0:.2f}'.format(currentProbability).rstrip('0').rstrip('.'))
            print(textIteration)
            
            #Add the new probability to the vector
            probabilities.append(currentProbability)
            
            #Checks if the algorithm is not improving. If so it stops the cycle.
            if iteration > 1 and currentProbability >= probabilities[iteration - 1]:
                #Break the cycle
                break
        
        # Corrects the probability of the FINAL state
        A[numStates - 2, numStates - 1] = 1 - A[numStates - 2].sum()
            
        #Updates the atrributes of the HMM object
        self.transMatrix = A
        self.emissionMatrix = B
        self.pi[:] = pi
               
        
    def calculateViterbi(self, obsSeq):
        """
        Finds the most probable sequences of states based on given a observation
        sequence using the Viterbi algorithm.
        
        Parameters
        ----------
        obsSeq : array of string
            Observation sequence

        Returns
        -------
        states : array of ints
            The pointers to the states of the sequence
        statesText : array of strings
            The names of the states of the sequence
        prob : float
            The probability of the sequence
        """
        
        #obtain the real names of the observations
        obsSeqPointers = self.translateEmissionToPointers(obsSeq)
        
        #Set the viterbi variables from the class attributes
        A = self.transMatrix
        O = self.emissionMatrix
        S = len(self.pi)
        pi = self.unstackMatrix(self.pi)
        
        M = len(obsSeqPointers)
        
        #Create the matrix with the forward probabilities
        fw = np.zeros((M, S))
        fw[:,:] = float('-inf')
        
        #Matrix for keeping the backpointers
        backpointers = np.zeros((M, S), 'int')
        
        #Initial step
        fw[0, :] = pi * O[:,obsSeqPointers[0]]

        #Inductive step
        for t in range(1, M):
            for s2 in range(S):
                for s1 in range(S):
                    score = fw[t-1, s1] * A[s1, s2] * O[s2, obsSeqPointers[t]]
                    if score > fw[t, s2]:
                        fw[t, s2] = score
                        backpointers[t, s2] = s1
        
        #normalize the probabilities
        reg_fw = np.zeros((M, S))
        reg_fw[:,:] = float('-inf')
        for i in range(fw.shape[0]):
            reg_fw[i,:] = fw[i,:] / sum(fw[i,:])
        
        #now follow backpointers to resolve the state sequence
        #and Calculate the probability of the sequence
        ss = []
        ss.append(np.argmax(fw[M-1,:]))
        prob = 1
        for i in range(M-1, 0, -1):
            prob = prob * reg_fw[i, ss[-1]]
            ss.append(backpointers[i, ss[-1]])
            
        
        #get the names of the states ordered
        states = list(reversed(ss))
        statesText = self.translateStatesPointers(states)
   
        #returns the pointers to the states of the sequence, the names of the 
        #states of the sequence and the probability of the sequence
        return states, statesText,  prob


    def unstackMatrix(self, matrix):
        """
        Unstacks a matrix so it fixes its format (hotfix)        
        
        Parameters
        ----------
        matrix : numpy matrix
            The matrix to unstack
        
        Returns
        -------
        unstackedMatrix : numpy matrix
            The matrix unstacked
        """
        return np.squeeze(np.asarray(matrix))
        
    
    def getOrderedDictionaryKeysByValue(self, dictionary):
        """
        Gets the keys of a dictionary ordered by its values      
        
        Parameters
        ----------
        dictionary : dictionary<K,V>
            The dictionary from which the keys are going to be ordered
        
        Returns
        -------
        sortedKeys : array of <K>
            The keys of the dictionary ordered by its values
        """
        return sorted(dictionary, key = dictionary.get)
        
    
    def drawFrom(self, probabilities):
        """
        Given an array of probabilities, this function returns a random index of the array    
        
        Parameters
        ----------
        probabilities : array of floats
            The array with probabilities to draw one of its indexes
        
        Returns
        -------
        index : int
            A random index (based on the probabilities) of the probabilities array.
        """
        return np.random.choice(len(probabilities), 1, p=probabilities)[0]


    def translateStatesPointers(self, statesPointers):
        """
        Function that translates an array of pointers to the respective state names using
        the statesDict dictionary.
        
        Parameters
        ----------
        statesPointers : array of ints
            The array of pointers to the respective state names
        
        Returns
        -------
        translation : array of strings
            The respective names of the states based on the given pointers
        """
        
        translation = []
        orderedStates = self.getOrderedDictionaryKeysByValue(self.statesDict)
        for statePointer in statesPointers:
            translation.append(orderedStates[statePointer])
        return translation
        
    
    def translateEmissionPointers(self, emissionsPointers):
        """
        Function that translates an array of pointers to the respective emission names using
        the statesDict dictionary.
        
        Parameters
        ----------
        emissionsPointers : array of ints
            The array of pointers to the respective emission names
        
        Returns
        -------
        translation : array of strings
            The respective names of the emissions based on the given pointers
        """
        translation = []
        orderedEmissions = self.getOrderedDictionaryKeysByValue(self.emissionsDict)
        for emissionPointer in emissionsPointers:
            translation.append(orderedEmissions[emissionPointer])
        return translation
    
    
    def translateEmissionToPointers(self, emissions):
        """
        Function that translates an array of emissions names to the respective 
        pointers (indexes) using the statesDict dictionary.
        
        Parameters
        ----------
        emissions : array of strings
            The names of the emissions
        
        Returns
        -------
        emissionPointers : array of ints
            The array of pointers to the given emission names
        """
        
        emissionPointers = []
        for em in emissions:
            emissionPointers.append(self.emissionsDict[em])
        return emissionPointers
    
    
if __name__ == '__main__':
    """Main function of the HMM. If this is invoked it runs a HmmCmd"""
    from HmmCmd import HmmCmd
    import sys
    args = sys.argv
    HmmCmd(args)
   
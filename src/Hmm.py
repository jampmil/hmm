# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:01:40 2016

@author: Jose
"""

import numpy as np

class Hmm():
    
    INIT_STATE = 'INIT'
    FINAL_STATE = 'FINAL'
    
    def __init__(self, transMatrix, pi, emissionMatrix, statesDict = {}, emissionsDict = {}):
        self.transMatrix = transMatrix
        self.pi = pi
        self.emissionMatrix = emissionMatrix
        self.statesDict = statesDict
        self.emissionsDict = emissionsDict
        
        
    def printHmm(self):
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

    
    def generateSequences(self, num):
        
        sequences = []

        for i in range(num):
            #Current Sequence and emissions Pointers
            seqPointers = []
            emiPointers = []
            
            #end state pointer
            endStatePointer = self.statesDict[self.FINAL_STATE]
            
            #get random state from the Pi vector
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
                    
            #print seqPointers, self.translateStatesPointers(seqPointers)
            #print emiPointers, self.translateEmissionPointers(emiPointers)
            
            #Append the found sequence to the final sequences
            sequences.append(self.translateEmissionPointers(emiPointers))
            
        return sequences
        
    def calculateForward(self, observation):
        """Calculates the alpha matrix using the forward algorithm"""
        
        # set up
        k = len(observation)
        (n, m) = self.transMatrix.shape
        fw = np.zeros( (n, k + 1) )
        
        fw[:, 0] = self.pi
        for obs_ind in xrange(k):
            f_row_vec = np.matrix(fw[:, obs_ind])
            fw[:, obs_ind + 1] = f_row_vec * \
                               np.matrix(self.transMatrix) * \
                               np.matrix(np.diag(self.emissionMatrix[:, self.emissionsDict[observation[obs_ind]]]))
            #Normalize
            fw[:, obs_ind + 1] = fw[:, obs_ind + 1] / np.sum(fw[:, obs_ind+1])
        
        return fw
        
    def calculateBackward(self, observation):
        """Calculates the beta matrix using the forward algorithm"""
        
        # set up
        k = len(observation)
        (n, m) = self.transMatrix.shape
        bw = np.zeros( (n ,k + 1) )
        
        bw[:,-1] = 1.0
        for obs_ind in xrange(k, 0, -1):
            b_col_vec = np.matrix(bw[:, obs_ind]).transpose()
            bw[:, obs_ind - 1] = (np.matrix(self.transMatrix) * \
                                np.matrix(np.diag(self.emissionMatrix[:, self.emissionsDict[observation[obs_ind - 1]]])) * \
                                b_col_vec).transpose()
            #Normalize
            bw[:, obs_ind - 1] = bw[:, obs_ind-1] / np.sum(bw[:, obs_ind-1])
        
        return bw
    
    def calculateGamma(self, alpha, beta):
        gamma = np.array(alpha) * np.array(beta)
        gamma = gamma / np.sum(gamma, 0)
        return gamma
        
    def calculateXi(self, alpha, beta, observation):
        n = self.emissionMatrix.shape[0]
        m = len(observation)
        xi = np.zeros((n, n, m - 1));
        for t in np.arange(m - 1):
            for i in np.arange(n):
                d1 = alpha[i, t] * self.transMatrix[i, :]
                d2 = d1.T * self.emissionMatrix[:, self.emissionsDict[observation[t+1]]]
                d3 = d2 * np.asmatrix(beta[:, t + 1]).T
                xi[i, :, t] = self.unstackMatrix(d3)
       
        return xi.sum(axis=2)
        
    def calculateBaumWelch(self, observations, maxIter = 10):
        
#        print 'Pi', self.pi
#        print 'A', self.transMatrix
#        print 'B', self.emissionMatrix
       
        #set the variables of the algorithm
        nStates = self.emissionMatrix.shape[0]
        criterion = 0.0001
        
        pi = self.pi
        A = np.asarray(self.transMatrix)        
        B = np.asarray(self.emissionMatrix)
        
        #Hotfix: Adds a FINAL Emission for the FINAL state with prob 1
        zeros = np.zeros(B.shape[0])
        zeros = np.matrix(zeros).T
        B = np.append(B, zeros, axis=1)
        B[B.shape[0] - 1, B.shape[1] -1] = 1
        B = np.asarray(B)
        
        #Hotfix: add the FINAL emission to the dict
        self.emissionsDict[self.FINAL_STATE] = len(self.emissionsDict.keys())
        
        prevA, prevB, prevpi = A,B,pi
        
        done = False
        countIter = 0
        while not done:
            
            for observation in observations:
                #Hotfix: add the FINAL emission to all observations
                if not observation[-1] == self.FINAL_STATE:
                    observation.append(self.FINAL_STATE)
                
                
                pointersObservation = self.translateEmissionToPointers(observation)
                pointersObservation = np.asarray(pointersObservation)
                nSamples = len(pointersObservation)
                
    
                
                # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
                # Initialize alpha
                alpha = np.zeros((nStates,nSamples))
                c = np.zeros(nSamples) #scale factors
                alpha[:,0] = pi.T * B[:,pointersObservation[0]]

                denC = np.sum(alpha[:,0])               
                c[0] = 1.0 
                if denC > 0:
                    c[0] = 1.0 / denC
                
                alpha[:,0] = c[0] * alpha[:,0]
                # Update alpha for each observation step
                for t in range(1,nSamples):
                    alpha[:,t] = np.dot(alpha[:,t-1].T, A).T * B[:,pointersObservation[t]]
                    denC = np.sum(alpha[:,t])
                    c[t] = 1.0
                    if denC > 0:
                        c[t] = 1.0/denC
                    alpha[:,t] = c[t] * alpha[:,t]
    
    #            print 'ALPHAS'
    #            print alpha.shape
    #            print self.calculateForward(observation).shape
    
                # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
                # Initialize beta
                beta = np.zeros((nStates,nSamples))
                beta[:,nSamples-1] = 1
                beta[:,nSamples-1] = c[nSamples-1] * beta[:,nSamples-1]
                # Update beta backwards from end of sequence
                for t in range(len(pointersObservation)-1,0,-1):
                    beta[:,t-1] = np.dot(A, (B[:,pointersObservation[t]] * beta[:,t]))
                    beta[:,t-1] = c[t-1] * beta[:,t-1]
    
    #            print 'BETAS'
    #            print beta.shape
    #            print self.calculateBackward(observation).shape
    
                xi = np.zeros((nStates,nStates,nSamples-1));
                for t in range(nSamples-1):
                    denom = np.dot(np.dot(alpha[:,t].T, A) * B[:,pointersObservation[t+1]].T,
                                   beta[:,t+1])
                    for i in range(nStates):
                        numer = alpha[i,t] * A[i,:] * B[:,pointersObservation[t+1]].T * \
                                beta[:,t+1].T
                        xi[i,:,t] = numer 
                        if denom > 0 :
                            xi[i,:,t] = numer / denom
      
                # gamma_t(i) = P(q_t = S_i | O, hmm)
                gamma = np.squeeze(np.sum(xi,axis=1))
                # Need final gamma element for new B
                prod =  (alpha[:,nSamples-1] * beta[:,nSamples-1]).reshape((-1,1))
                
                den_prod = np.sum(prod)
                if den_prod > 0:
                    gamma = np.hstack((gamma,  prod / den_prod)) #append one more to gamma!!!
                else:
                    gamma = np.hstack((gamma,  prod)) #append one more to gamma!!!
    
                newpi = gamma[:,0]
    
                num_A = np.sum(xi,2)
                den_A = np.sum(gamma[:,:-1], axis=1).reshape((-1,1))
                newA = num_A
                for i in range(len(num_A)):
                    if den_A[i] > 0:
                        newA[i] = num_A[i]/ den_A[i]
                    else:
                        newA[i] = num_A[i]
                #newA = num_A / den_A
                
                newB = np.copy(B)
    
                numLevels = B.shape[1]
                sumgamma = np.sum(gamma,axis=1)                
                for lev in range(numLevels):
                    mask = pointersObservation == lev
                    num_mask = np.sum(gamma[:,mask],axis=1)
                    #newB[:,lev] = num_mask / sumgamma
                    for i in range(len(sumgamma)):
                        if sumgamma[i] > 0:
                            newB[i,lev] = num_mask[i]/ sumgamma[i]
                        else:
                            newB[i,lev] = num_mask[i]
    
            if np.max(abs(prevpi - newpi)) < criterion and \
                   np.max(abs(prevA - newA)) < criterion and \
                   np.max(abs(prevB - newB)) < criterion:
                done = 1;
#                print 'MIN CHANGE REACHED'
            elif countIter >= maxIter:
                done = 1
#                print 'MAX ITER REACHED'
            else:
                countIter += 1
            
#            print 'NUM_ITER=' + str(countIter)
  
            prevA[:],prevB[:],prevpi[:] = newA,newB,newpi

        #Hotfix: Remove last column of the new emission matrix and remove the FINAL state from the dict
        newB = newB[:,:-1]
        del self.emissionsDict[self.FINAL_STATE]
        
        
#        print 'Pi', newpi
#        print 'A', newA
#        print 'B', newB
        
        self.pi[:] = newpi
        self.transMatrix[:] = newA
        self.emissionMatrix[:] = newB
        
        

        
#        n = self.emissionMatrix.shape[0]
#        numObs = len(self.emissionsDict.keys()) #change to numEm
#
#        for iteration in np.arange(maxIter):
#            b_bar_den = np.ones(n)
#            a_bar_den = np.ones(n)
#            a_bar_num = np.ones((n, n))
#            pi_bar = np.ones(n)
#            b_bar_num = np.ones((n, numObs))
#            for obs in observations:
#                # alpha, log_prob_obs, c = self.forward(obs)
#                # beta = self.backward(obs, c)
#                alpha = self.calculateForward(obs)
#                beta = self.calculateBackward(obs)
#                gamma = self.calculateGamma(alpha, beta)
#                xi = self.calculateXi(alpha, beta, obs)
#                
##                print "alpha"
##                print alpha
##                print "beta"    
##                print beta
##                print "gamma"
##                print gamma
##                print "xi"
##                print xi
#                
#                
#                
#                #log_likelihood += log_prob_obs
#                #T = len(obs)
#                #index_obs = self._index_observations(obs)
#
#                pi_bar *= gamma[:, 0]
#
#                b_bar_den *= gamma.sum(1)
#                a_bar_den *= gamma.sum(1)
#              
#                
#                a_bar_num *= xi.sum(1) #??
#                orderedEmissions = self.getOrderedDictionaryKeysByValue(self.emissionsDict)
#                B_bar = np.zeros([n, numObs])
#                for k in np.arange(numObs):
#                    indicator = np.array(
#                        [orderedEmissions[k] == x for x in obs])
#                    B_bar[:, k] = gamma.T[indicator, :].sum(0)
#                # b_bar_num += w * B_bar
#                b_bar_num += B_bar
#            
#            #Update Pi
#            self.pi = pi_bar / np.sum(pi_bar)
#            #update A
#            A_bar = np.zeros(self.transMatrix.shape)
#            #A_bar[0, :] = pi_bar / np.sum(pi_bar)
#            for i in np.arange(1, n - 1):
#                A_bar[i, :] = a_bar_num[i, :] / a_bar_den[i]
#            # A_bar[self.N-2, self.N-1] = 1 - A_bar[self.N-2].sum() # correct
#            # final silent state
#            self.transMatrix = A_bar
#            # update B
#            for i in np.arange(1, n - 1):
#                if b_bar_den[i] > 0:
#                    b_bar_num[i, :] = b_bar_num[i, :] / b_bar_den[i]
#                else:
#                    b_bar_num[i, :] = b_bar_num[i, :]
#            self.emissionMatrix = b_bar_num
#            
#            print 'Pi', self.pi
#            print 'A', self.transMatrix
#            print 'B', self.emissionMatrix
            
            
        #self.transMatrix[n - 2, n - 1] = 1 - \
        #    self.transMatrix[n - 2].sum()  # correct final silent state

        
        
        
    def calculateViterbi(self, observation):
        
        #obtain the real names of the observations
        observations = []
        for ob in observation:
            observations.append(self.emissionsDict[ob])
        
        #Set the viterbi variables from the class attributes
        A = self.transMatrix
        O = self.emissionMatrix
        S = len(self.pi)
        pi = self.unstackMatrix(self.pi)
        
        M = len(observations)
        
        #Create the matrix with the forward probabilities
        fw = np.zeros((M, S))
        fw[:,:] = float('-inf')
        
        #Matrix for keeping the backpointers
        backpointers = np.zeros((M, S), 'int')
        
        #Initial step
        fw[0, :] = pi * O[:,observations[0]]

        #Inductive step
        for t in range(1, M):
            for s2 in range(S):
                for s1 in range(S):
                    score = fw[t-1, s1] * A[s1, s2] * O[s2, observations[t]]
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
        prod = 1
        for i in range(M-1, 0, -1):
            prod = prod * reg_fw[i, ss[-1]]
            ss.append(backpointers[i, ss[-1]])
            
        
        #get the names of the states ordered
        tokensText = self.translateStatesPointers(list(reversed(ss)))
   
        #returns the pointers to the states of the sequence, the names of the 
        #states of the sequence and the probability of the sequence
        return list(reversed(ss)), tokensText,  prod


    def unstackMatrix(self, matrix):
        return np.squeeze(np.asarray(matrix))
        
    def getOrderedDictionaryKeysByValue(self, dictionary):
        return sorted(dictionary, key = dictionary.get)
        
    def drawFrom(self, probabilities):
           return np.random.choice(len(probabilities), 1, p=probabilities)[0]

    def translateStatesPointers(self, statesPointers):
        translation = []
        orderedStates = self.getOrderedDictionaryKeysByValue(self.statesDict)
        for statePointer in statesPointers:
            translation.append(orderedStates[statePointer])
        return translation
        
    def translateEmissionPointers(self, emissionsPointers):
        translation = []
        orderedEmissions = self.getOrderedDictionaryKeysByValue(self.emissionsDict)
        for emissionPointer in emissionsPointers:
            translation.append(orderedEmissions[emissionPointer])
        return translation
        
    def translateEmissionToPointers(self, emissions):
        emissionPointers = []
        for em in emissions:
            emissionPointers.append(self.emissionsDict[em])
        return emissionPointers
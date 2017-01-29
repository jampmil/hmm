import numpy as np
from operator import itemgetter


class HMM:
    """
    Hidden Markov Model Class
    """

    def __init__(self, A=None, B=None, name_states=None, name_observations=None):
        self.A = np.array(A)
        self.B = np.array(B)
        self.name_states = name_states
        self.name_observations = name_observations
        self.N = len(name_states)
        self.M = len(name_observations)

    def _log_probabilities(self, X):
        lp = np.zeros((X.shape))
        lp[X > 0] = np.log(X[X > 0])
        lp[X == 0] = float('-inf')
        return lp

    def _index_observations(self, obs):
        y = []
        for o in obs:
            y.append(self.name_observations.index(o))
        return y

    def gen_sequence(self, num_sequences=1):
        """
        Generate sequences of emissions given the current model
        :param num_sequences: number of sequences to generate
        :return: array with the sequences generated
        """
        def _draw_from(probabilities):
            return np.random.choice(len(probabilities), 1, p=probabilities)[0]

        sequences = []
        for i in np.arange(0, num_sequences):
            states = []
            observations = []
            states.append(_draw_from(self.A[0]))
            observations.append(_draw_from(self.B[states[0]]))
            t = 1
            state = _draw_from(self.A[states[t - 1]])
            while state < len(self.name_states) - 1:
                states.append(state)
                observations.append(_draw_from(self.B[state]))
                t += 1
                state = _draw_from(self.A[states[t - 1]])
            sequences.append(itemgetter(*observations)(self.name_observations))
        return sequences

    def viterbi(self, observations):
        """
        Viterbi
        :param observations: array of sequences of observations
        :return: array of sequences of states with their probability
        """
        sequences = []
        Alog = self._log_probabilities(self.A)
        Blog = self._log_probabilities(self.B)
        for obs in observations:
            T = len(obs)
            delta = np.zeros([self.N, T])
            psi = np.zeros([self.N, T])
            index_obs = self._index_observations(obs)
            delta[:, 0] = Alog[0, :] + Blog[:, index_obs[0]]
            psi[:, 0] = 0
            for t in np.arange(1, T):
                tmp = np.array([delta[:, t - 1]]).T + Alog
                delta[:, t] = np.amax(tmp, axis=0) + Blog[:, index_obs[t]]
                psi[:, t] = np.argmax(tmp, axis=0)
            max_prob = np.exp(np.max(delta[:, T - 1]))
            sequence = np.array([np.argmax(delta[:, T - 1])], dtype='int')
            for t in np.arange(T - 1, 0, -1):
                sequence = np.insert(sequence, 0, psi[sequence[0], t])
            seq = [self.name_states[i] for i in sequence]
            sequences.append((seq, max_prob))
        return sequences

    def forward(self, observations):
        # sequences = []
        # for i in np.arange(len(observations)):
        T = len(observations)
        c = np.zeros([T])
        alpha = np.zeros([self.N, T])
        index_obs = self._index_observations(observations)
        alpha[:, 0] = self.A[0, :] * self.B[:, index_obs[0]]
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= c[0]
        for t in np.arange(1, T):
            alpha[:, t] = np.dot(
                alpha[:, t - 1], self.A) * self.B[:, index_obs[t]]
            c[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] *= c[t]
        log_prob_obs = -(np.sum(np.log(c)))
        return alpha, log_prob_obs, c
        # sequences.append((alpha, log_prob_obs, c))
        # return sequences

    def backward(self, observations, c):
        # sequences = []
        # for i in np.arange(len(observations)):
        T = len(observations)
        index_obs = self._index_observations(observations)
        beta = np.zeros([self.N, T])
        beta[:, T - 1] = c[T - 1]
        for t in np.arange(T - 1, 0, -1):
            beta[:, t - 1] = np.dot(self.A,
                                    self.B[:, index_obs[t]] * beta[:, t])
            beta[:, t - 1] *= c[t - 1]
        return beta
        #     sequences.append(beta)
        # return sequences

    def _forward_backward(self, observations):
        T = len(observations)
        c = np.zeros([T])
        alpha = np.zeros([self.N, T])
        beta = np.zeros([self.N, T])
        index_obs = self._index_observations(observations)
        alpha[:, 0] = self.A[0, :] * self.B[:, index_obs[0]]
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= c[0]
        for t in np.arange(1, T):
            alpha[:, t] = np.dot(alpha[:, t - 1], self.A) * \
                self.B[:, index_obs[t]]
            c[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] *= c[t]
        log_prob_obs = -(np.sum(np.log(c)))
        beta[:, T - 1] = c[T - 1]
        for t in np.arange(T - 1, 0, -1):
            beta[:, t - 1] = np.dot(self.A,
                                    self.B[:, index_obs[t]] * beta[:, t])
            beta[:, t - 1] *= c[t - 1]
        return alpha, beta, log_prob_obs

    def baum_welch(self, observations, max_iter=20):
        """
        Baum-Welch
        :param observations: array of sequences of observations
        :param max_iter: number of max iterations
        :return: HMM instance trained
        """
        log_likelihoods = []
        for epoch in np.arange(max_iter):
            log_likelihood = 0
            b_bar_den = np.zeros([self.N])
            a_bar_den = np.zeros([self.N])
            a_bar_num = np.zeros([self.N, self.N])
            pi_bar = np.zeros([self.N])
            b_bar_num = np.zeros([self.N, self.M])
            for obs in observations:
                # alpha, log_prob_obs, c = self.forward(obs)
                # beta = self.backward(obs, c)
                alpha, beta, log_prob_obs = self._forward_backward(obs)
                log_likelihood += log_prob_obs
                T = len(obs)
                index_obs = self._index_observations(obs)
                # w = 1.0 / -(log_prob_obs + np.log(T))
                gamma_raw = alpha * beta
                gamma = gamma_raw / gamma_raw.sum(0)
                # pi_bar += w * gamma[:, 0]
                pi_bar += gamma[:, 0]
                # b_bar_den += w * gamma.sum(1)
                b_bar_den += gamma.sum(1)
                # a_bar_den += w * gamma[:, :T-1].sum(1)
                # a_bar_den += gamma[:, :T-1].sum(1)
                # a_bar_den += w * gamma.sum(1)
                a_bar_den += gamma.sum(1)
                xi = np.zeros([self.N, self.N, T - 1])
                for t in np.arange(T - 1):
                    for i in np.arange(self.N):
                        xi[i, :, t] = alpha[i, t] * self.A[i, :] * \
                            self.B[:, index_obs[t + 1]] * beta[:, t + 1]
                # a_bar_num += w * xi[:, :, :T - 1].sum(2)
                # a_bar_num += w * xi.sum(2)
                a_bar_num += xi.sum(2)
                B_bar = np.zeros([self.N, self.M])
                for k in np.arange(self.M):
                    indicator = np.array(
                        [self.name_observations[k] == x for x in obs])
                    B_bar[:, k] = gamma.T[indicator, :].sum(0)
                # b_bar_num += w * B_bar
                b_bar_num += B_bar
            # update A
            A_bar = np.zeros([self.N, self.N])
            A_bar[0, :] = pi_bar / np.sum(pi_bar)
            for i in np.arange(1, self.N - 1):
                A_bar[i, :] = a_bar_num[i, :] / a_bar_den[i]
            # A_bar[self.N-2, self.N-1] = 1 - A_bar[self.N-2].sum() # correct
            # final silent state
            self.A = A_bar
            # update B
            for i in np.arange(1, self.N - 1):
                if b_bar_den[i] > 0:
                    b_bar_num[i, :] = b_bar_num[i, :] / b_bar_den[i]
                else:
                    b_bar_num[i, :] = b_bar_num[i, :]
            self.B = b_bar_num
            print(log_likelihood)
            log_likelihoods.append(log_likelihood)
            if epoch > 1 and log_likelihood >= log_likelihoods[epoch - 1]:
                print('not improving')
                break
        self.A[self.N - 2, self.N - 1] = 1 - \
            self.A[self.N - 2].sum()  # correct final silent state
        return self, epoch + 1


if __name__ == '__main__':


    states = ['INIT', 'Onset', 'Mid', 'End', 'FINAL']
    observations = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    A = [[0, 1, 0, 0, 0],
         [0, 0.3, 0.7, 0, 0],
         [0, 0, 0.9, 0.1, 0],
         [0, 0, 0, 0.4, 0.6],
         [0, 0, 0, 0, 0]]
    B = [[0, 0, 0, 0, 0, 0, 0],
         [0.5, 0.2, 0.3, 0, 0, 0, 0],
         [0, 0, 0.2, 0.7, 0.1, 0, 0],
         [0, 0, 0, 0.1, 0, 0.5, 0.4],
         [0, 0, 0, 0, 0, 0, 0]]
    A_ini = [[0, 1, 0, 0, 0],
             [0, 0.5, 0.5, 0, 0],
             [0, 0, 0.5, 0.5, 0],
             [0, 0, 0, 0.5, 0.5],
             [0, 0, 0, 0, 0]]
    B_ini = [[0, 0, 0, 0, 0, 0, 0],
             [0.33, 0.33, 0.33, 0, 0, 0, 0],
             [0, 0, 0.33, 0.33, 0.33, 0, 0],
             [0, 0, 0, 0.33, 0, 0.33, 0.33],
             [0, 0, 0, 0, 0, 0, 0]]
    seq = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
           ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6']]
    seq1 = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
            ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6'],
            ['C1', 'C2', 'C3', 'C6']]
    seq_train = [['C1', 'C4', 'C6', 'C7', 'C6'],
                 ['C1', 'C5', 'C7'],
                 ['C2', 'C1', 'C3', 'C1', 'C1', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4',
                  'C4', 'C5', 'C5', 'C3', 'C4', 'C4', 'C3', 'C3', 'C5', 'C7', 'C6', 'C7'],
                 ['C2', 'C1', 'C5', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C6'],
                 ['C3', 'C4', 'C7', 'C7'],
                 ['C2', 'C3', 'C4', 'C3', 'C4', 'C3', 'C6', 'C6'],
                 ['C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4',
                  'C4', 'C5', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C3', 'C6'],
                 ['C3', 'C3', 'C1', 'C4', 'C3', 'C5',
                  'C4', 'C4', 'C4', 'C4', 'C4', 'C6'],
                 ['C2', 'C3', 'C5', 'C4', 'C7'],
                 ['C2', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C6'],
                 ['C1', 'C4', 'C6', 'C7'],
                 ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4',
                  'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C3', 'C5', 'C7'],
                 ['C2', 'C4', 'C7'],
                 ['C3', 'C4', 'C7'],
                 ['C2', 'C1', 'C4', 'C4', 'C4', 'C4', 'C3',
                  'C4', 'C4', 'C4', 'C3', 'C4', 'C5', 'C7'],
                 ['C1', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C6', 'C7'],
                 ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C6'],
                 ['C1', 'C3', 'C4', 'C4', 'C3', 'C5', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C4', 'C4', 'C3',
                  'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C4', 'C7'],
                 ['C1', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4',
                  'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C6', 'C7'],
                 ['C3', 'C4', 'C6'],
                 ['C2', 'C4', 'C4', 'C5', 'C6'],
                 ['C2', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C7'],
                 ['C1', 'C5', 'C4', 'C4', 'C7'],
                 ['C1', 'C4', 'C5', 'C4', 'C6'],
                 ['C3', 'C4', 'C5', 'C4', 'C4', 'C3', 'C4', 'C5', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C5', 'C4', 'C5', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C6'],
                 ['C3', 'C3', 'C3', 'C4', 'C3', 'C5', 'C3', 'C4',
                  'C5', 'C4', 'C4', 'C4', 'C4', 'C6', 'C4'],
                 ['C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C6'],
                 ['C3', 'C2', 'C1', 'C1', 'C2', 'C3', 'C3', 'C6', 'C6', 'C6'],
                 ['C3', 'C4', 'C4', 'C4', 'C3', 'C6', 'C6'],
                 ['C2', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C5',
                  'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C3', 'C5', 'C3', 'C4', 'C3', 'C7', 'C6'],
                 ['C1', 'C4', 'C5', 'C4', 'C6', 'C6'],
                 ['C3', 'C1', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4',
                  'C3', 'C3', 'C3', 'C4', 'C4', 'C4', 'C7', 'C4'],
                 ['C1', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7', 'C4'],
                 ['C3', 'C1', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C7'],
                 ['C2', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C5', 'C4', 'C5', 'C4',
                  'C4',
                  'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3',
                  'C6'],
                 ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C6'],
                 ['C2', 'C1', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4',
                  'C4', 'C4',
                  'C4', 'C3', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4',
                  'C4'],
                 ['C1', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C6', 'C6'],
                 ['C1', 'C4', 'C5', 'C4'],
                 ['C2', 'C1', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C4', 'C3', 'C6'],
                 ['C1', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4',
                  'C4', 'C3', 'C4', 'C4', 'C5', 'C5', 'C3', 'C4', 'C7', 'C7'],
                 ['C2', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C4', 'C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C7', 'C6', 'C7'],
                 ['C2', 'C3', 'C6'],
                 ['C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C3', 'C4', 'C6', 'C6', 'C4', 'C7'],
                 ['C2', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C7'],
                 ['C3', 'C1', 'C1', 'C1', 'C2', 'C4', 'C4', 'C4',
                  'C3', 'C4', 'C4', 'C6', 'C6', 'C6', 'C4', 'C7'],
                 ['C3', 'C2', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C5', 'C4', 'C5', 'C3', 'C4',
                  'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C3', 'C6', 'C4'],
                 ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C5', 'C4', 'C4', 'C5', 'C4', 'C4',
                  'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C7', 'C7', 'C7'],
                 ['C1', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C5', 'C6', 'C6', 'C6'],
                 ['C2', 'C4', 'C4', 'C4', 'C6', 'C7'],
                 ['C3', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4',
                  'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C7', 'C7', 'C6'],
                 ['C3', 'C4', 'C4', 'C3', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C7'],
                 ['C3', 'C4', 'C4', 'C6', 'C7'],
                 ['C1', 'C4', 'C4', 'C4', 'C7', 'C6'],
                 ['C3', 'C1', 'C3', 'C3', 'C4', 'C4', 'C4', 'C6'],
                 ['C3', 'C1', 'C4', 'C3', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C3', 'C4', 'C6'],
                 ['C3', 'C1', 'C1', 'C1', 'C1', 'C3', 'C7', 'C6'],
                 ['C1', 'C4', 'C6', 'C6'],
                 ['C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C3', 'C3', 'C6'],
                 ['C1', 'C3', 'C5', 'C5', 'C4', 'C3', 'C4', 'C4', 'C4', 'C7'],
                 ['C3', 'C3', 'C7', 'C7'],
                 ['C1', 'C3', 'C3', 'C4', 'C3', 'C5',
                  'C5', 'C3', 'C4', 'C3', 'C4', 'C6'],
                 ['C1', 'C3', 'C6', 'C6'],
                 ['C1', 'C4', 'C5', 'C4', 'C4', 'C6', 'C6'],
                 ['C2', 'C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4'],
                 ['C3', 'C5', 'C4', 'C3', 'C3', 'C4', 'C4',
                  'C4', 'C3', 'C4', 'C5', 'C4', 'C7'],
                 ['C3', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C7', 'C7'],
                 ['C2', 'C1', 'C3', 'C3', 'C4', 'C3', 'C4',
                  'C3', 'C4', 'C4', 'C4', 'C4', 'C6'],
                 ['C2', 'C1', 'C4', 'C5', 'C4', 'C4', 'C7', 'C4', 'C6', 'C6'],
                 ['C1', 'C1', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C7'],
                 ['C3', 'C1', 'C4', 'C5', 'C4', 'C6'],
                 ['C1', 'C3', 'C4', 'C3', 'C5', 'C4', 'C3', 'C4',
                  'C4', 'C4', 'C4', 'C3', 'C6', 'C6', 'C6'],
                 ['C2', 'C1', 'C3', 'C1', 'C1', 'C3', 'C4', 'C4', 'C4', 'C6', 'C6'],
                 ['C1', 'C2', 'C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3',
                  'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7', 'C6'],
                 ['C3', 'C3', 'C4', 'C4', 'C6'],
                 ['C1', 'C4', 'C4', 'C4', 'C5', 'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4',
                  'C4', 'C3', 'C4', 'C3', 'C4',
                  'C3', 'C4', 'C4', 'C4', 'C4', 'C5', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C3', 'C4',
                  'C4', 'C4', 'C4', 'C6'],
                 ['C1', 'C3', 'C4', 'C4', 'C6', 'C6', 'C7'],
                 ['C2', 'C3', 'C4', 'C4', 'C4', 'C4', 'C7'],
                 ['C1', 'C3', 'C4', 'C4', 'C6', 'C6'],
                 ['C1', 'C3', 'C3', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4',
                  'C4', 'C5', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C3', 'C7'],
                 ['C2', 'C1', 'C4', 'C3', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4',
                  'C3', 'C4', 'C4', 'C3', 'C4', 'C5', 'C4', 'C5', 'C6', 'C7'],
                 ['C1', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C3', 'C3', 'C5', 'C5', 'C5', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C7'],
                 ['C1', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4',
                  'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C7'],
                 ['C2', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4',
                  'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C7', 'C6'],
                 ['C2', 'C4', 'C3', 'C5', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C5', 'C4', 'C3', 'C3', 'C3', 'C6'],
                 ['C1', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C6', 'C6'],
                 ['C3', 'C3', 'C4', 'C4', 'C4', 'C4',
                  'C5', 'C4', 'C4', 'C4', 'C5', 'C7'],
                 ['C3', 'C4', 'C4'],
                 ['C1', 'C4', 'C4', 'C7'],
                 ['C1', 'C3', 'C4', 'C5', 'C3', 'C4', 'C7', 'C6'],
                 ['C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7', 'C4', 'C7'],
                 ['C1', 'C3', 'C4', 'C4'],
                 ['C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C6'],
                 ['C3', 'C3', 'C5', 'C4', 'C5', 'C4'],
                 ['C1', 'C1', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C3', 'C4', 'C6', 'C7', 'C6', 'C6', 'C7'],
                 ['C3', 'C3', 'C3', 'C4', 'C4', 'C7', 'C6'],
                 ['C3', 'C1', 'C3', 'C4', 'C4', 'C3', 'C3', 'C6'],
                 ['C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C6', 'C7'],
                 ['C2', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
                  'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C7'],
                 ['C1', 'C3', 'C2', 'C4', 'C4', 'C6', 'C7', 'C7']]
    h = HMM(A, B, states, observations)
    h1 = HMM(A_ini, B_ini, states, observations)
    h2, ep = h1.baum_welch(seq_train)
    print 'ep', ep
    print('A', h2.A)
    print('B', h2.B)
    print(h2.A.sum(1))
    print(h2.B.sum(1))
    
    print ('B_ini', np.matrix(B_ini))
    print 'B_ini', np.matrix(B_ini).shape
                            
    print 'shape A,B', np.matrix(A_ini).shape, np.matrix(B_ini).shape
    print "N,M", h1.N, h1.M
    print type(h1.A)
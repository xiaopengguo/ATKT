# -*- coding: utf-8 -*-
import numpy as np

class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, maxstep, name="data"):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen
        self.maxstep = maxstep

    def load_data(self, path):
        f_data = open(path, 'r')
        skill_data = []
        answer_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 3 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                mod = 0 if len(S) % self.maxstep == 0 else (self.maxstep - len(S) % self.maxstep)
                
                for i in range(len(S)):
                    skill_data.append(int(int(S[i])-1))
                    answer_data.append(int(A[i]))
                for j in range(mod):
                    skill_data.append(-1)
                    answer_data.append(-1)
        f_data.close()
        return np.array(skill_data).astype(np.int).reshape([-1, self.maxstep]), np.array(answer_data).astype(np.int).reshape([-1, self.maxstep])

class PID_DATA(object):
    def __init__(self, n_question, seqlen, separate_char, maxstep, name="data"):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen
        self.maxstep = maxstep

    def load_data(self, path):
        f_data = open(path, 'r')
        skill_data = []
        answer_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 4 == 2:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                    
                mod = 0 if len(S) % self.maxstep == 0 else (self.maxstep - len(S) % self.maxstep)
                
                for i in range(len(S)):
                    skill_data.append(int(int(S[i])-1))
                    answer_data.append(int(A[i]))
                for j in range(mod):
                    skill_data.append(-1)
                    answer_data.append(-1)
        f_data.close()
        return np.array(skill_data).astype(np.int).reshape([-1, self.maxstep]), np.array(answer_data).astype(np.int).reshape([-1, self.maxstep])
import numpy as np

'''
Perform xor on each bit of the binary representation of a decimal number
11: [1 0 1 1] -> 1+0+1+1 -> 1
'''
def xor_bits(decimal): 
    result=0
    while decimal>0:
        result=result^(decimal&1)
        decimal=decimal>>1
    return result

'''
Concatenate two bits and return the decimal representation
ex: [c1 c0]=[1 1] -> 3
'''
def binary_to_decimal(c1,c0):
    return (c1<<1)|c0

class poly2trellis:
    def __init__(self,k,n,v,polynomial):
        self.k_=k
        self.n_=n
        self.v_=v
        self.p0_octal=polynomial[0]
        self.p1_octal=polynomial[1]
        self.numInputSymbols=2**self.k_
        self.numOutputSymbols=2**self.n_
        self.numStates=2**self.v_
        self.nextStates=self.computeNextStates()
        self.outputs=self.computeOutputs()

    def computeNextStates(self):
        state_matrix=np.zeros((self.numStates,self.numInputSymbols),dtype=int)
        for i in range(0,self.numStates):

            # current state (row)
            current_state=i  

            # input = 0
            next_state=current_state>>1
            state_matrix[i][0]=next_state

            # input = 1
            next_state=(current_state>>1)+2**(self.v_-1)         
            state_matrix[i][1]=next_state

        return state_matrix
    
    def computeOutputs(self):
        p0_decimal=int(str(self.p0_octal),8) #11
        p1_decimal=int(str(self.p1_octal),8) #15

        output_matrix=np.zeros((self.numStates,self.numInputSymbols), dtype=int)
        for current_state in range(0,self.numStates): #state: 0~7
            # input (u0): 0
            input_state=current_state
            c1=xor_bits(np.bitwise_and(p0_decimal,input_state))
            c0=xor_bits(np.bitwise_and(p1_decimal,input_state))
            output_matrix[current_state][0]=binary_to_decimal(c1,c0)

            # input (u0): 1
            input_state=current_state+2**self.v_
            c1=xor_bits(np.bitwise_and(p0_decimal,input_state))
            c0=xor_bits(np.bitwise_and(p1_decimal,input_state))
            output_matrix[current_state][1]=binary_to_decimal(c1,c0)
            
        return output_matrix
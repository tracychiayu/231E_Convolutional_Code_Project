import numpy as np

def ZTCC_dmin(K,v,trellis):
    total_number_of_states=trellis.numStates  # 2**v
    total_number_of_stages=K+v+1
    total_number_of_outputs=2*(K+v)
    metrics=5000*np.ones((total_number_of_states,total_number_of_stages),dtype=int)
    metrics[0][0]=0  # Hamming weight of total outputs
    number_of_path=np.zeros((total_number_of_states,total_number_of_stages),dtype=int)
    number_of_path[0][0]=1
    # output_mapping=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=int)
    output_mapping=np.array([0,1,1,2],dtype=int)

    # stage 0~(K+v-1):
    for current_stage in range(K+v):
        for current_state in range(total_number_of_states):

            # input: 0
            # input: 1 (cannot go from zero state to zero state for the first step)
            if not (current_stage==0 and current_state==0):
                next_state=trellis.nextStates[current_state][0]
                output_weight=output_mapping[trellis.outputs[current_state][0]]
                total_weight=metrics[current_state][current_stage]+output_weight

                if metrics[next_state][current_stage+1] > total_weight:
                    metrics[next_state][current_stage+1]=total_weight
                    number_of_path[next_state][current_stage+1]=number_of_path[current_state][current_stage]

                elif metrics[next_state][current_stage+1] == total_weight:
                    number_of_path[next_state][current_stage+1] += number_of_path[current_state][current_stage]

            # input: 1
            next_state=trellis.nextStates[current_state][1]
            output_weight=output_mapping[trellis.outputs[current_state][1]]
            total_weight=metrics[current_state][current_stage]+output_weight

            if metrics[next_state][current_stage+1] > total_weight:
                metrics[next_state][current_stage+1]=total_weight
                number_of_path[next_state][current_stage+1]=number_of_path[current_state][current_stage]
            elif metrics[next_state][current_stage+1] == total_weight:
                number_of_path[next_state][current_stage+1] += number_of_path[current_state][current_stage]

            if (metrics[0][current_stage+1] ==  np.min(metrics[:,current_stage+1])) and metrics[0][current_stage+1]!=5000:
                dmin=metrics[0][current_stage+1]  
                
    '''             
    Calculate total number of nearest number after we copmlete the entire matrix
    Calculate the total number of nearest neighbors (NN) by summing the values in the first row of 'number_of_path' 
    starting from the column where the first occurrence of dmin is found, and including all columns where the value is equal to dmin.
    '''  
    # Find the index where dmin first appears
    for i in range(total_number_of_stages):
        if metrics[0][i]==dmin:
            dmin_index=i
            break

    # Count total number of nearest neighbor
    NN=0
    for i in range(dmin_index,total_number_of_stages):
        NN=NN+number_of_path[0][i]

    return dmin,NN

def TBCC_dmin(K,v,trellis):
    total_number_of_states=trellis.numStates  # 2**v
    total_number_of_stages=K+1
    total_number_of_outputs=2*K
    output_mapping=np.array([0,1,1,2],dtype=int)

    # store dmin and NN values for different start/end state cases 
    dmins=np.zeros(total_number_of_states,dtype=int)
    NNs=np.zeros(total_number_of_states,dtype=int)

    # i: start and end state
    for i in range(total_number_of_states):

        # create new matrices
        metrics=5000*np.ones((total_number_of_states,total_number_of_stages),dtype=int)
        number_of_path=np.zeros((total_number_of_states,total_number_of_stages),dtype=int)
        metrics[i][0]=0
        number_of_path[i][0]=1
        
        # 從這開始改8/3
        # fill out 'metrics' and 'number_of_path' matrices
        for current_stage in range(total_number_of_stages-1):  # K (stage: 0 ~ (K-1))
            for current_state in range(total_number_of_states): # 2^v states

                # input: 0
                if not (i==0 and current_state==0 and current_stage==0):
                    next_state=trellis.nextStates[current_state][0]
                    output_weight=output_mapping[trellis.outputs[current_state][0]]
                    total_weight=metrics[current_state][current_stage]+output_weight

                    if metrics[next_state][current_stage+1] > total_weight:
                        metrics[next_state][current_stage+1]=total_weight
                        number_of_path[next_state][current_stage+1]=number_of_path[current_state][current_stage]
                    
                    elif metrics[next_state][current_stage+1] == total_weight:
                        number_of_path[next_state][current_stage+1] += number_of_path[current_state][current_stage]
                        
                # input: 1
                next_state=trellis.nextStates[current_state][1]
                output_weight=output_mapping[trellis.outputs[current_state][1]]
                total_weight=metrics[current_state][current_stage]+output_weight

                if metrics[next_state][current_stage+1] > total_weight:
                    metrics[next_state][current_stage+1]=total_weight
                    number_of_path[next_state][current_stage+1]=number_of_path[current_state][current_stage]
                
                elif metrics[next_state][current_stage+1] == total_weight:
                    number_of_path[next_state][current_stage+1] += number_of_path[current_state][current_stage]


        # dmin: look at the ith row and the last column to find the value
        dmins[i]=metrics[i][total_number_of_stages-1]

        # Calculation of Nearest Neighbor (NN)
        # case that start & end at all_zero state
        if i==0: 
            # find the column that dmin first appears
            for col in range(total_number_of_stages):
                if metrics[0][col] == dmins[0]:
                    idx=col
                    break

            NNs[0]+=(K-idx+1)*number_of_path[0][idx]

            for col in range(idx+1,total_number_of_stages):
                if number_of_path[0][col]>number_of_path[0][col-1]:
                    NNs[0]+=(K-col+1)*(number_of_path[0][col]-number_of_path[0][col-1])

        else:
            NNs[i]=number_of_path[i][total_number_of_stages-1]

    dmin=np.min(dmins)

    NN=0
    for i in range(total_number_of_states):
        if dmins[i]==dmin:
            NN+=NNs[i]

        
    return dmin,NN







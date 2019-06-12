# Import your files here...
import numpy as np
import re

def read_file(State_File, Symbol_File,Q3):
    if Q3:
        transition_dict = {}
        data_state = []
        with open(State_File,'r') as state_file:
            for line in state_file.readlines():
                data_state.append(line.strip())
        #print(data)
        N = int(data_state[0])
        state_dict = {}
        for n in range(N):
            state_dict[data_state[n+1]]=n
        init_matrix = np.matrix
        transition_matrix = np.zeros((N,N),float)
        for n in range(N+1,len(data_state)):
            ll = data_state[n].split()
            transition_matrix[int(ll[0]),int(ll[1])]=int(ll[2])
        for i in range(len(transition_matrix)-1):
            f_dict=dict()
            f_dict_new=dict()
            Sum=sum(transition_matrix[i])
            for j in range(len(transition_matrix[i])):
                value=transition_matrix[i,j]
                if value not in f_dict.keys():
                    f_dict[value]=1
                else:
                    f_dict[value]+=1
            for j in range(len(transition_matrix[i])):
                key=sorted(f_dict.keys())
                if key.index(transition_matrix[i,j])==len(key)-1:
                    nb=key.index(transition_matrix[i,j])
                else:
                    nb=key.index(transition_matrix[i,j])+1
                Nkey=key[nb]
                p = (transition_matrix[i,j]+1)*(f_dict[Nkey]/f_dict[transition_matrix[i][j]])
                transition_matrix[i][j]= p/Sum
            if i == state_dict['BEGIN']:
                init_matrix = transition_matrix[i,:]

        #print(transit_matrix,init_matrix)
        data_symbol = []
        emission_dict = {}
        symbol_dict = {}
        with open(Symbol_File, 'r') as symbol_file:
            for line in symbol_file.readlines():
                data_symbol.append(line.strip())
        #print(data_symbol)
        M = int(data_symbol[0])
        for n in range(M):
            symbol_dict[data_symbol[n+1]]=n
        #print(symbol_dict)
        emission_matrix = np.zeros((N,M+1),float)
        
        for n in range(M+1,len(data_symbol)):
            ll = data_symbol[n].split()
            emission_matrix[int(ll[0]),int(ll[1])]=int(ll[2])
        for i in range(len(emission_matrix)-2):
            f_dict=dict()
            f_dict_new=dict()
            Sum=sum(emission_matrix[i])
            for j in range(len(emission_matrix[i])):
                value=emission_matrix[i,j]
                if value not in f_dict.keys():
                    f_dict[value]=1
                else:
                    f_dict[value]+=1
                #a=sorted(f_dict.items(),key= lambda item :item[0])
            for j in range(len(emission_matrix[i])):
                key=sorted(f_dict.keys())
                if key.index(emission_matrix[i,j])==len(key)-1:
                    nb=key.index(emission_matrix[i,j])
                else:
                    nb=key.index(emission_matrix[i,j])+1
                Nkey=key[nb]
                p = (emission_matrix[i,j]+1)*(f_dict[Nkey]/f_dict[emission_matrix[i][j]])
                emission_matrix[i][j]= p/Sum
    else:
        transition_dict = {}
        data_state = []
        with open(State_File,'r') as state_file:
            for line in state_file.readlines():
                data_state.append(line.strip())
        #print(data)
        N = int(data_state[0])
        state_dict = {}
        for n in range(N):
            state_dict[data_state[n+1]]=n
        #print(state_dict)
        #transition_matrix = np.matrix
        for n in range(N+1,len(data_state)):
            ll = data_state[n].split()
            ld = []
            for e in ll:
                ld.append(int(e))

            transition_dict.setdefault(ld[0],{})
            transition_dict[ld[0]][ld[1]] = ld[2]
        transition_matrix = np.zeros((N,N),float)
        init_matrix = np.matrix
        for i in range(N):
            if i != state_dict['END']:
                if i in transition_dict.keys():
                    sum_value = sum(transition_dict[i].values())
                    #print(sum_value)
                else:
                    sum_value = 0
                for j in range(N):
                    if j == state_dict['BEGIN']:
                        continue
                    if j in transition_dict[i].keys() and i in transition_dict.keys():
                        transition_matrix[i,j] = (1 + transition_dict[i][j])/(sum_value + N - 1)
                    else:
                        transition_matrix[i,j] = 1/(sum_value + N - 1)
                if i == state_dict['BEGIN']:
                    init_matrix = transition_matrix[i,:]

        #print(transit_matrix,init_matrix)
        data_symbol = []
        emission_dict = {}
        symbol_dict = {}
        with open(Symbol_File, 'r') as symbol_file:
            for line in symbol_file.readlines():
                data_symbol.append(line.strip())
        #print(data_symbol)
        M = int(data_symbol[0])
        for n in range(M):
            symbol_dict[data_symbol[n+1]]=n
        #print(symbol_dict)
        emission_matrix = np.zeros((N,M+1),float)

        for n in range(M+1,len(data_symbol)):
            ll = data_symbol[n].split()
            ld = []
            for e in ll:
                ld.append(int(e))
            emission_dict.setdefault(ld[0],{})
            emission_dict[ld[0]][ld[1]] = ld[2]
        for i in range(N):
            if state_dict['BEGIN'] != i and state_dict['END'] != i:
                if i in emission_dict.keys():
                    sum_value = sum(emission_dict[i].values())
                else:
                    sum_value = 0
                for j in range(M+1):
                    if j in emission_dict[i].keys() and i in emission_dict.keys():
                        emission_matrix[i,j] = (1+emission_dict[i][j])/(sum_value+M+1)
                    else:
                        emission_matrix[i,j] = 1/(sum_value+M+1)
        
    return N,M,state_dict,symbol_dict,init_matrix,transition_matrix,emission_matrix

def compute_probability(N, state_dict, symbol_dict, init_matrix, transition_matrix, emission_matrix, query):
    init_state = []
    for i in range(N):
        if query[0] in symbol_dict.keys():
            init_state.append(init_matrix[i] * emission_matrix[i, symbol_dict[query[0]]])
        else:
            init_state.append(init_matrix[i] * emission_matrix[i, symbol_dict['UNK']])
    #print(init_state)
    state_record = {}
    for n in range(1,len(query)):
        if n == 1:
            state_record[n] = {'cur_list':[],'cur_state':[]}
            ld = []
            lid = []
            for i in range(len(init_state)):
                ll = []
                for m in range(N):
                    ll.append(init_state[m] * transition_matrix[m, i])
                mm = max(ll)
                index = ll.index(mm)
                lid.append(index)
                if query[n] in symbol_dict.keys():
                    ld.append(mm*emission_matrix[i,symbol_dict[query[n]]])
                else:
                    ld.append(mm*emission_matrix[i,symbol_dict['UNK']])
            state_record[n]['cur_list'].append(ld)
            state_record[n]['cur_state'].append(lid)
        else:
            state_record[n] = {'cur_list': [], 'cur_state': []}
            ld = []
            lid = []
            for i in range(len(init_state)):
                ll = []
                for m in range(N):
                    ll.append(state_record[n-1]['cur_list'][0][m] * transition_matrix[m, i])
                mm = max(ll)
                index = ll.index(mm)
                lid.append(index)
                if query[n] in symbol_dict.keys():
                    ld.append(mm*emission_matrix[i,symbol_dict[query[n]]])
                else:
                    ld.append(mm*emission_matrix[i,symbol_dict['UNK']])
            state_record[n]['cur_list'].append(ld)
            state_record[n]['cur_state'].append(lid)
    log_prob = []
    for m in range(N):
        log_prob.append(state_record[len(query)-1]['cur_list'][0][m] * transition_matrix[m, -1])
    log_value = max(log_prob)
    log_value = np.log(log_value)
    #print(log_value)
    path = []
    path.append(state_dict['END'])
    final_choice = state_record[len(query)-1]['cur_list'][0]
    #print(state_record,final_choice)
    pos = final_choice.index(max(final_choice))
    path.append(pos)
    #print(pos,path)
    for n in range(len(query)-1,0,-1):
        pos = state_record[n]['cur_state'][0][pos]
        path.append(pos)
    path.append(state_dict['BEGIN'])
    path1 = path[::-1]
    path1.append(log_value)
    #print(path1)
    return path1

def compute_top_k(N, state_dict, symbol_dict, init_matrix, transition_matrix, emission_matrix, query, k):
    dp_collection = [[[[[], 0.0] for _ in range(k)] for _ in range(len(query)+2)] for _ in range(N)]
    dp_collection[state_dict['BEGIN']][0][0] = [[], 1]
    all_symbols = symbol_dict.keys()
    for i in range(N):
        if query[0] in all_symbols:
            dp_collection[i][1][0][1] = init_matrix[i] * emission_matrix[i, symbol_dict[query[0]]]
        else:
            dp_collection[i][1][0][1] = init_matrix[i] * emission_matrix[i, symbol_dict['UNK']]
        dp_collection[i][1][0][0].append(state_dict['BEGIN'])

    for j in range(2, len(query) + 1):
        for i in range(N):
            collection = []
            for state in range(N):
                for x in range(k):
                    if query[j - 1] in all_symbols:
                        collection.append((dp_collection[state][j - 1][x][0] + [state], dp_collection[state][j - 1][x][1] * transition_matrix[state, i] * emission_matrix[i, symbol_dict[query[j - 1]]]))
                    else:
                        collection.append((dp_collection[state][j - 1][x][0] + [state], dp_collection[state][j - 1][x][1] * transition_matrix[state, i] * emission_matrix[i, symbol_dict['UNK']]))
            collection = sorted(collection, key=lambda x: x[1], reverse=True)
            for x in range(k):
                dp_collection[i][j][x][0].extend(collection[x][0])
                dp_collection[i][j][x][1] = collection[x][1]

    end_matrix = transition_matrix[:, state_dict['END']]
    for i in range(N):
        for x in range(k):
            dp_collection[i][len(query) + 1][x][1] = end_matrix[i] * dp_collection[i][len(query)][x][1]
            LL = dp_collection[i][len(query)][x][0] + [i]
            dp_collection[i][len(query) + 1][x][0] += LL

    result = []
    for i in range(N):
        result.extend(dp_collection[i][len(query) + 1])
    result = sorted(result, key=lambda x: x[1], reverse=True)
    top_k_list = []
    for i in range(k):
        path = result[i][0] + [state_dict['END']] + [np.log(result[i][1])]
        top_k_list.append(path)
    return top_k_list


# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    q1_list = []
    split_pattern = r"[0-9A-Za-z\.!@#\$%\\\^\*+=\{\}\[\]\"'<>~\·`\?:;|]+|[,&-/()]"
    N, M, state_dict, symbol_dict, init_matrix, transition_matrix, emission_matrix = read_file(State_File, Symbol_File,False)

    with open(Query_File, 'r') as query_file:
        for line in query_file.readlines():
            query = re.compile(split_pattern).findall(line)
            symbol_dict["UNK"] = M
            path = compute_probability(N,state_dict,symbol_dict,init_matrix,transition_matrix,emission_matrix,query)
            q1_list.append(path)
    return q1_list

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    q2_list = []
    split_pattern = r"[0-9A-Za-z\.!@#\$%\\\^\*+=\{\}\[\]\"'<>~\·`\?:;|]+|[,&-/()]"
    N,M,state_dict,symbol_dict,init_matrix,transition_matrix,emission_matrix = read_file(State_File, Symbol_File,False)

    with open(Query_File, 'r') as query_file:
        for line in query_file.readlines():
            query = re.compile(split_pattern).findall(line)
            symbol_dict["UNK"] = M
            q2_list += compute_top_k(N,state_dict,symbol_dict,init_matrix,transition_matrix,emission_matrix,query,k)
    return q2_list

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    q3_list = []
    split_pattern = r"[0-9A-Za-z\.!@#\$%\\\^\*+=\{\}\[\]\"'<>~\·`\?:;|]+|[,&-/()]"
    N, M, state_dict, symbol_dict, init_matrix, transition_matrix, emission_matrix = read_file(State_File, Symbol_File,True)

    with open(Query_File, 'r') as query_file:
        for line in query_file.readlines():
            query = re.compile(split_pattern).findall(line)
            symbol_dict["UNK"] = M
            path = compute_probability(N,state_dict,symbol_dict,init_matrix,transition_matrix,emission_matrix,query)
            q3_list.append(path)
    return q3_list

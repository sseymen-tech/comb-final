import networkx as nx
from networkx import bipartite    
import numpy as np
import random


def plotGraph(graph,ax,title):    
    pos=[(ii[1],ii[0]) for ii in graph.nodes()]
    pos_dict=dict(zip(graph.nodes(),pos))
    nx.draw(graph,pos=pos_dict,ax=ax,with_labels=True)
    ax.set_title(title)
    return   

def create_training(filename, seed, node_num, sparsity_array):
    
    edges = []
    
    for i in range(node_num):
        edges.append([str(i+1),str(i+1)])
        for j in range(i+1,node_num):
            random_num = np.random.uniform(0,1)
            sparsity_param = (sparsity_array[i]+sparsity_array[j])/2
            if random_num < sparsity_param:  ###no self-loops
#                print()
                edges.append([str(j+1),str(i+1)])
                
    edge_num = len(edges)
    
    save_file = filename + '_'+str(node_num) + '_'+str(edge_num) + '_ALL(0.2)' + 'train_seed'+str(seed)+'.txt'
    file1 = open(save_file,"w") 
    file1.write(str(node_num))
    file1.write(' ')
    file1.write(str(node_num))
    file1.write(' ')
    file1.write(str(edge_num))
    file1.write('\n')     
    for xx in edges:
        file1.write(str(int(xx[0])))
        file1.write(' ')
        file1.write(str(int(xx[1])))
        file1.write('\n') 
    file1.close() 
    
    np.random.shuffle(edges)
    return edges, node_num, edge_num 

def read_edges(file):
    flagged = 0
    edges = []
    all_sets = []
    with open(file) as fp: 
        for line in fp: 
#            print(line)
            line.replace("\n","")
#            print(line)
            if flagged == 1:
                nums = line.split(" ")
                if int(nums[0]) != int(nums[1]): ###SO DELETE THE SELF LOOPS BASICALLY
#                    if set([ int(nums[0]), int(nums[1])]) in all_sets:
#                        print('oof')
#                    else:
                    edges.append([ int(nums[0]), int(nums[1])])
#                        all_sets.append(set([ int(nums[0]), int(nums[1])]))
#                        print(all_sets)
                
                
            elif line[0] != '%':
                flagged = 1
                nums = line.split(" ")
                node_num = max(int(nums[0]),int(nums[1]))
                edge_num = int(nums[2])
             
    
    return edges, node_num, edge_num
    
if __name__=='__main__':    
    #---------------Construct the graph---------------
#    count_x = -1
#    sparsities = [0.00005]

#    sparsities = [(900/662)/662*2,(4660/800)/800*2,(4195/1089)/1089*2,(46/39)/39*2,(8271/5300)/5300*2,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.0007]
#    for node_num in [662,800,1089,39,5300,153,62,72,198,2680,472,406,9129,1440,4008,258,20545]:
#        count_x +=1
#        spars_no = sparsities[count_x]
    node_num=72
    for seed in range(0,1):
#            print(node_num)
#            np.random.seed(seed)
        g=nx.Graph()
        filename = 'C:/Users/sseym/Desktop/files/'
    #    dataname = '662_bus/662_bus.mtx'
    #    dataname = 'bcspwr10/bcspwr10.mtx'
        
    
    #    dataname = 'G15/G15.mtx'
        dataname = 'dwt_72/dwt_72.mtx'
        
        ###After this do poli again! - bayer and the other 4
        edges, node_num, edge_num = read_edges(filename+dataname)  ###for reading the data
        sparsity_array = [0.036]*node_num # In (0,1)
    #        for i in range(node_num):
    #            sparsity_array[i][i]=1
    
#        edges, node_num, edge_num = create_training(filename, seed, node_num, sparsity_array) ###create new data function
    
        for ii in edges:
            g.add_node(ii[0])
            g.add_node(ii[1])
    #                print('a',ii[0],ii[1])
    
        g.add_edges_from(edges)
    
        #---------------Use maximal_matching---------------
        match=nx.max_weight_matching(g,True)    
        g_match=nx.Graph()
        for ii in match:
            g_match.add_edge(ii[0],ii[1])
    
    #    print('nx.degree_g',nx.degree(g))
    #        print('nx.degree_g_match',nx.degree(g_match))
    #            print('nx.edges_g',nx.number_of_edges(g))
    #            print('nx.edges_g_match',nx.number_of_edges(g_match))
        aa_matrix = nx.convert_matrix.to_numpy_matrix(g)
    #    for ix in range(node_num):
    #        aa_matrix[ix,ix]=0    
        bb_matrix = np.zeros((node_num,node_num))
        np.savetxt(str(node_num)+'G15_numpy_g_original'+'.txt', aa_matrix, fmt='%d')
#        np.savetxt(str(node_num)+'G15_numpy_g_original' +'.txt', aa_matrix, fmt='%d') 
        for (iy,ik) in g_match.edges():
    #                print('A')
    #                print(iy,ik)
            bb_matrix[int(iy)-1,int(ik)-1] = 1.
            bb_matrix[int(ik)-1,int(iy)-1] = 1.
    #            print('nx.g_match_edges',g_match.edges())
        
#        np.savetxt(str(node_num)+'G15_numpy_g_match_original' +'.txt', bb_matrix, fmt='%d')
        np.savetxt(str(node_num)+'G15_numpy_g_match_original' +'.txt', bb_matrix, fmt='%d')
    #    nx.edges(g_match)
        print('nx.g_edges',nx.number_of_edges(g))
        print('nx.g_match_edges',len(g_match.edges()))
        #-----------------------Plot-----------------------
    #    import matplotlib.pyplot as plt
    #    fig=plt.figure(figsize=(10,8))
    #
    ##    ax1=fig.add_subplot(2,2,1)
    ##    plotGraph(g,ax1,'Graph')
    #
    #    ax2=fig.add_subplot(2,2,2)
    #    plotGraph(g_match,ax2,'nx.max_weight_matching()')
    #
    #
    #    plt.show()
    #        save_file = filename + '_'+str(node_num) + '_'+str(edge_num) +'_ALL(0.2)'+'label_seed'+str(seed)+'.txt'
    #        file1 = open(save_file,"w") 
    #        for xx in g_match.edges():
    #            file1.write(str(int(xx[0])))
    #            file1.write(' ')
    #            file1.write(str(int(xx[1])))
    #            file1.write('\n') 
    #        file1.close() 
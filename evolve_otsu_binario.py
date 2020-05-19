#!/usr/bin/env python
# coding: utf-8


# Imports
from __future__ import print_function
import os

import random
import numpy as np
from skimage.io import imread, imread_collection, imsave
#from scipy.misc import imsave as save
import random
from skimage.filters import median,threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from skimage import util 
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square, erosion, dilation
from skimage.measure import regionprops
from skimage.color import lab2rgb
from functools import reduce
import math as m
import glob
from sklearn.model_selection import train_test_split
#get_ipython().run_line_magic('matplotlib', 'inline')


#################################    SETTINGS    #######################################
#############################################################################################

#Here you inform the default settings of the algorithm:

FOLDER_RESULTS = 'novoteste500'  # THIS IS THE RESULTS FOLDER, YOU HAVE TO CREATE IT TO EXECUTE
NUMBER_INDIVIDUALS = 500  # Individuals number
NUMBER_GENERATIONS = 100 # Number of generations

NUMBER_JOBS = 16 # Number of processor cores for the joblib

#############################################################################################
#############################################################################################

# Loading images and masks
imagens = imread_collection('PH2Dataset/PH2 Dataset images/*/*'+'_Dermoscopic_Image/*')
mascaras_medico = imread_collection('PH2Dataset/PH2 Dataset images/*/*'+'_lesion/*')


# Dividing into training and testing
x_train, x_test, y_train, y_test = train_test_split(imagens, mascaras_medico, test_size = 0.2)


############### initial population


# some necessary functions
# Function that converts the binary array to a binary string
def arraybinario_to_string(binario):
    str_bin = str(binario).strip('[]').replace(" ", "")
    return str_bin

# Function that converts the binary string to real
def converte_decimal(binario):
    alelo_dec = int(binario, 2)
    if alelo_dec > 9999:
        alelo_dec = 9999
    
    return float("0." + str(alelo_dec))

# decimal function to binary
def d2b(n):
    if n == 0:
        return ''
    else:
        return d2b(n//2) + str(n%2)
    

# My real to binary function
def real_to_binario(real):
    real = int(real*10000)
    return d2b(real)


def binary_initialization(population, parameters, n_bits):
    """
        Parameters
        __________
            population:
            parameters:
            n_bits:
        Return
        ______
            Returns a population
        Example
        _______
            # Binary initialization
            father = binary_initialization(20, 3, 14)
            mother = binary_initialization(20, 3, 14)
    """

    populacao_bin = np.around(np.random.random((population, parameters, n_bits)), decimals=0).astype(int)
    
    # Adding the initial population of individuals r, g and b
    # enabled = 0.9999
    ativado = [1,0,0,1,1,1,0,0,0,0,1,1,1,1]
    desativado = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #red
    populacao_bin[0][0] = ativado
    populacao_bin[0][1] = desativado
    populacao_bin[0][2] = desativado
    #green
    populacao_bin[1][0] = desativado
    populacao_bin[1][1] = ativado
    populacao_bin[1][2] = desativado
    #blue
    populacao_bin[2][0] = desativado
    populacao_bin[2][1] = desativado
    populacao_bin[2][2] = ativado
    #rgb
    populacao_bin[3][0] = ativado
    populacao_bin[3][1] = ativado
    populacao_bin[3][2] = ativado
    
    return populacao_bin

def populacao_binaria_to_real(population_binaria,first=False):
    
    if first == True:
        # Define a matrix of N dimension what get the population with random values
        population = np.zeros((population_binaria.shape[0], 3))
    else:
        population = np.zeros((population_binaria.shape[0], 4))
    
    for i in range(population_binaria.shape[0]):
        for j in range(3):
            # Set a random value uniform (between begin_gene and end_begin) 
            
            array_binario = arraybinario_to_string(population_binaria[i, j])
            population[i, j] = converte_decimal(array_binario)

    return population


def populacao_real_to_binaria(populacao_real):
    linhas = populacao_real.shape[0]
    alelos = 3
    n_bits = 14

    populacao_bin = np.zeros((linhas,alelos,n_bits)).astype(int)

    for i,pop_bin in enumerate(populacao_bin):
        for alelo in range(alelos):
            alelo_atual = populacao_real[i,alelo]
            binario_atual = real_to_binario(alelo_atual)
            binario_atual = list(binario_atual)

            inicio = n_bits-len(binario_atual)

            populacao_bin[i,alelo,inicio:] = binario_atual
    return populacao_bin


################################ Evaluation

########### My RGB to Gray function
def rgb2grayrafa(x,y,z,imagem):
    r, g, b = imagem[:,:,0], imagem[:,:,1], imagem[:,:,2]
    imagem_gray = x * r + y * g + z * b
    
    #imagem_gray = minmax_scale(imagem_gray, feature_range=(0,255))
    imagem_gray /= (imagem_gray.max()/1.0)
    return imagem_gray

########### Segmentation Function
def segmentaMelanomaComBorda(imagem):

    im = imagem

    # Blurs the image to help segment hair
    img_med = median(im,np.ones((100,100)))
    # takes the best otsu threshold
    limiar = threshold_otsu(im)
    # Removes loose pieces
    img_bin = closing(im > limiar, square(5))
    # Invert black and white so you can remove the edges
    inverted_img = util.invert(img_bin)
    # Remove the border
    cleared = clear_border(inverted_img)
    
    # Condition to identify images that have become completely black
    if cleared.mean() < 0.1:
        cleared = inverted_img
        
    # read the chewing labels
    labels  =  label ( cleared ) 

    # take the regions
    regions = regionprops(labels)
    
    try:
        # select only the largest region
        maior_region = reduce((lambda x, y: x if x['Area']>y['Area'] else y), regions)
    except:
        
        return cleared

    # generates the mask of the largest region
    mask = np.zeros(labels.shape)
    mask[maior_region['coords'][:,0],maior_region['coords'][:,1]]=1

    return mask

# Calculates dice
def melhorDiceNovo(mascaras_minha,mascara_medico):
    inte =  np.logical_and(mascara_medico, mascaras_minha)
    uniao = np.logical_or(mascara_medico, mascaras_minha)
    resultado = np.sum(inte)/np.sum(uniao)
    
    return resultado


#######################################################################
########################## usando joblib
import joblib 
def processa_rgb_segmenta_dice(x,y,z,imagem,mascara):
    im = rgb2grayrafa(x,y,z,imagem)
    segmentacaocom = segmentaMelanomaComBorda(im)
    dicecom = melhorDiceNovo(segmentacaocom,mascara)
    
    return dicecom
    
    
########################## using joblib
def ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico):
    resultado = joblib.Parallel(n_jobs=NUMBER_JOBS)(joblib.delayed(processa_rgb_segmenta_dice)(x,y,z,imagens[nr],mascaras_medico[nr]) for nr in range(len(imagens)))
          
    return np.array(resultado).mean()
#######################################################################



# variable to control the print of which individual started the process resumption
retomada = False

# Evaluation function of individuals (fitness evaluation)
def Avaliacao(populacao,geracao,imagens,mascaras_medico,n_genes):
    
    global retomada
    
    populacao_nova = np.zeros((populacao.shape[0],n_genes+1))
    
    # If it is the initial population, there is no index, if it is any other generation, it is the index.
    if populacao.shape[1] == 3:
        for i,p in enumerate(populacao):
            x = p[0]
            y = p[1]
            z = p[2]
            
            populacao_nova[:,:-1] = populacao
            dice = ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico)
            print('##### Fim do Individuo ',i,' - X ',x,' Y ',y,' Z ',z,' - Dice: ',dice)
            populacao_nova[i,-1] = dice
            # Saving the index of each individual who ran on the network
            np.savetxt(FOLDER_RESULTS+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
    
    # When it is not the initial population, but that of a read CSV
    else:
        
        # Check if all individuals have already been evaluated
        # all evaluated
        if populacao.shape[0] == np.count_nonzero(populacao[:,-1]):
            populacao_nova[:,:-1] = populacao[:,:-1]
            for i,p in enumerate(populacao):
                x = p[0]
                y = p[1]
                z = p[2]

                dice = ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico)
                print('##### Fim do Individuo ',i,' - X ',x,' Y ',y,' Z ',z,' - Dice: ',dice)
                populacao_nova[i,-1] = dice
                # Saving the index of each individual who ran on the network
                np.savetxt(FOLDER_RESULTS+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
        
        # not all were evaluated or none
        else:
            populacao_nova = populacao
            for i,p in enumerate(populacao):
                x = p[0]
                y = p[1]
                z = p[2]
                
                # condition to calculate only those with a 0.0 index
                if populacao_nova[i,-1] == 0.0:
                    # Condition to show which individual you took back
                    if retomada == True:
                        print("##################  Retomando processo do individuo ", i)
                        retomada = False
                        
                    dice = ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico)
                    print('##### Fim do Individuo ',i,' - X ',x,' Y ',y,' Z ',z,' - Dice: ',dice)
                    populacao_nova[i,-1] = dice

                    # Saving the index of each individual who ran on the network
                    np.savetxt(FOLDER_RESULTS+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
                    retomada = False
 
    # sort by fitness
    populacao_nova = populacao_nova[populacao_nova[:,3].argsort()[::-1]]
    
    # Saving the new population with all the dices now sorted
    np.savetxt(FOLDER_RESULTS+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
    
    return populacao_nova, populacao_nova[:,3].mean()


############ Selection of father and mother for crossing

def individual_extractor(population, individual):
    """
        Parameters
        __________
            population: Population from which the individual will be extracted.
            individual: Individual who will be extracted.
        Return
        ______
            This function returns a single individual from the population.
        Example
        _______
            # Individual selector
            individual_extractor(population, individual)
    """

    #return np.array_str(population[individual, 1:-1])
    return population[individual]

def tournament_selection(population):
    """
        Parameters
        __________
            population: Population from which selection will be made.
            t_individuals: Number of individuals in the selected population.
        Return
        ______
            This function returns the selection of the best individuals in the population.
        Example
        _______
            # Selection by tournament
            selection = tournament_selection(population, 50)
    """
    t_individuals = population.shape[0]
    new_population = []
    random_selected = [t_individuals]
    count_individual = 0

    if t_individuals > len(population):
        print("Error: Total individuals of new population greater than current population.")
    else:
        while count_individual <= t_individuals:
            individual_selected = random.randint(0, len(population) - 1)
            if individual_selected not in random_selected:
                new_population.append(individual_extractor(population, individual_selected))
                random_selected.append(individual_selected)
                count_individual += 1
            if count_individual == t_individuals:
                break

    mothers = np.array(np.array_split(new_population, 2)[0])
    fathers = np.array(np.array_split(new_population, 2)[1])
    return mothers, fathers



########################### Crossover


"""
    Cite: EIBEN, Agoston E. et al. Introduction to evolutionary computing. Berlin: springer, 2003.
    4.2.2 Recombination for Binary Representation
    4.2 Binary Representation
        Uniform Crossover The previous two operators worked by dividing the
        parents into a number of sections of contiguous genes and reassembling them
        to produce offspring. In contrast to this, uniform crossover [422] works by
        treating each gene independently and making a random choice as to which
        parent it should be inherited from. This is implemented by generating a string
        of l random variables from a uniform distribution over [0,1]. In each position,
        if the value is below a parameter p (usually 0.5), the gene is inherited from
        the first parent; otherwise from the second. The second offspring is created
        using the inverse mapping.
"""


def uniform_crossover(father, mother):    
    children = np.zeros((father.shape[0]*2, father.shape[1], father.shape[2])).astype(int)
    index = 0
    for index_fm in range(father.shape[0]):
        for i in range(father.shape[1]):
            n_bits_arr = np.around(np.random.random(father.shape[2]), decimals=1)
            for j in range(father.shape[2]):
                if(n_bits_arr[j] < 0.5):
                    children[index][i][j] = father[index_fm][i][j]
                    children[index+1][i][j] = mother[index_fm][i][j]
                else:
                    children[index][i][j] = mother[index_fm][i][j]
                    children[index+1][i][j] = father[index_fm][i][j]
        index += 2
    return children



############################ Mutation


"""
    Cite: "Genetic Algorithms - Mutation."https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm. Acessado em 18 nov de 2019.
        Bit Flip Mutation - In this bit flip mutation, we select one or more random bits and flip them. 
        This is used for binary encoded GAs.
"""


def binary_bit_flip_mutation(individual):  
    """
        Parameters
        ----------
            individual: Receive an individul to be mutated
            
        Return 
        ------
            Return a individual with modificate bit
    """
    
    for i in range(individual.shape[0]):
        # Choose two alleles randomly
        bit = sorted(random.sample(range(1, individual[0].shape[0]), 1))
        individual[i][bit] = int(not(individual[i][bit]))
    return individual


def mutation_all(population):

    for individual in population:
        individual = binary_bit_flip_mutation(individual)
    return population


##################### NEW POPULATION

def novaPopulacao(population, childrens, elitism=20):
    
    nova_populacao = np.zeros((population.shape[0],population.shape[1],population.shape[2])).astype(int)
    
    tamanho_populacao = population.shape[0]
    # number of children generated and parents who will follow by elitism for the new population
    qtd_elitism = tamanho_populacao*elitism//100
    qtd_filhos = tamanho_populacao - qtd_elitism
    
    nova_populacao[:qtd_elitism] = population[:qtd_elitism]
    nova_populacao[qtd_elitism:] = childrens[:qtd_filhos]
    
    return nova_populacao


##################################### MAIN FUNCTION

def evoluirOtsu(imagens,mascaras_medico,n_indiduals,n_genes):
    global retomada
    print('######## inicio da evolução ########')
    dados_geracao = []
    
    #start binary and convert to real
    population_init = binary_initialization(population=n_indiduals, parameters=n_genes, n_bits=14)
    population_init = populacao_binaria_to_real(population_init,first=True)

    # Now let's see if there is a generation with individuals, if not there that uses the initial population
    csv_dados_evolucao = glob.glob(FOLDER_RESULTS+'/dados_evolucao-geracao*.csv')
    csv_dados_evolucao.sort()
    if len(csv_dados_evolucao) == 0:
        retomada = False
        print("################## Começando com a população inicial / geração 0")

        # Fitness
        geracao = 0
        #################### here the population already enters as real
        populacao, aptidao = Avaliacao(population_init,geracao,imagens,mascaras_medico, n_genes)

        print(" ######################################## ")
        print('Geração: ',geracao, ' / Media DSC: ', aptidao)
        print(" ######################################## ")

        dados_geracao.append(populacao)
    else:
        retomada = True
        # get the last csv
        population = np.genfromtxt(csv_dados_evolucao[-1],delimiter=',')
        geracao = len(csv_dados_evolucao)-1
        # if all have already been validated, increase the generation
        if population.shape[0] == np.count_nonzero(population[:,-1]):
            geracao +=1
        
        print("################## Retomando processo da geração", geracao)
        
        populacao, aptidao = Avaliacao(population,geracao,imagens,mascaras_medico, n_genes)
        print(" ######################################## ")
        print('Geração: ',geracao, ' / Media DSC: ', aptidao)
        print(" ######################################## ")
        
        dados_geracao.append(populacao)
        
    
    # stop condition
    while aptidao <= 0.99 and geracao < NUMBER_GENERATIONS: ########## HERE I DEFINE HOW MANY GENERATIONS WILL BE
        
        #################### here I take the population and convert it to the binary population
        ##### nao precisa ter o dice
        populacao_binaria = populacao_real_to_binaria(populacao)
        
        # Selection of fathers and mothers
        pai, mae = tournament_selection(populacao_binaria)
        
        # Crossover
        filhos = uniform_crossover(pai, mae)
        
        # mutating children
        filhos_mutados = mutation_all(filhos)

        # Conception of the new population
        nova_populacao_binaria = novaPopulacao(populacao_binaria, filhos_mutados, elitism=20)
        
        geracao += 1
        
        #################### here I convert the new population back to real
        ##### here you need to have the dice field, but with 0
        nova_populacao = populacao_binaria_to_real(nova_populacao_binaria,first=False)
        
        # New Fitness
        populacao, aptidao = Avaliacao(nova_populacao,geracao,imagens,mascaras_medico,n_genes)
        
        print(" ######################################## ")
        print('Geração: ',geracao, ' / Media DSC: ', aptidao)
        print(" ######################################## ")
        
        # population is the entire population of a generation
        dados_geracao.append(populacao)
    
    print('######## fim da evolução ########')
    # dados_geracao are all generations
    return dados_geracao



############################### HERE WHERE EXECUTION CALLS ###########################
#######################################################################################

# so ta valendo o parametro n_individuals, que é o numero de individuos
dados_geracao = evoluirOtsu(x_train,y_train,n_indiduals=NUMBER_INDIVIDUALS,n_genes=3)


#############################################################################################
#############################################################################################
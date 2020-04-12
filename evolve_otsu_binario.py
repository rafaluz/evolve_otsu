#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os

# Imports
import random
import numpy as np
from skimage.io import imread, imread_collection, imsave
from scipy.misc import imsave as save
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

#################################    CONFIGURAÇÕES    #######################################
#############################################################################################

PASTA = 'novoteste500'  # ESSA É A PASTA DOS RESULTADOS, TEM QUE CRIAR ELA PRA RODAR
QTD_INDIVIDUOS = 500  # NÚMERO DE INDIVIDUOS
NUMERO_GERACOES = 100 # NÚMERO DE GERACOES

NUMERO_JOBS = 16 # NÚMERO DE NÚCLEOS DO PROCESSADOR PARA O JOBLIB

#############################################################################################
#############################################################################################

# TODAS linux
imagens = imread_collection('PH2Dataset/PH2 Dataset images/*/*'+'_Dermoscopic_Image/*')
mascaras_medico = imread_collection('PH2Dataset/PH2 Dataset images/*/*'+'_lesion/*')


#x_train, x_val, y_train, y_val = train_test_split(imagens, mascaras_medico, test_size = 0.4)
x_train, x_test, y_train, y_test = train_test_split(imagens, mascaras_medico, test_size = 0.2)


##############3 # população inicial


# algumas funções necessárias
# Função que converte o array binario em uma string binaria
def arraybinario_to_string(binario):
    str_bin = str(binario).strip('[]').replace(" ", "")
    return str_bin

#Função que converte a string binaria para real
def converte_decimal(binario):
    alelo_dec = int(binario, 2)
    if alelo_dec > 9999:
        alelo_dec = 9999
    
    return float("0." + str(alelo_dec))

# função de decimal para binario
# def dec2bin(n):
#     b = ''
#     while n != 0:
#         b = b + str(n % 2)
#         n = int(n / 2)
#     return b[::-1]

# função de decimal para binario
def d2b(n):
    if n == 0:
        return ''
    else:
        return d2b(n//2) + str(n%2)
    

# Minha função de real para binario
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

    return np.around(np.random.random((population, parameters, n_bits)), decimals=0).astype(int)

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


#FUNÇÃO DE ROMUERE
# def bin_float(binary,mini,maxi):
#     b10 = int(binary,2)
#     x1 = mini+(maxi-mini)*b10/(2**len(binary)-1)
#     return x1



####################################33 # Avaliação

############################ alterado
def rgb2grayrafa(x,y,z,imagem):
    r, g, b = imagem[:,:,0], imagem[:,:,1], imagem[:,:,2]
    imagem_gray = x * r + y * g + z * b
    
    #imagem_gray = minmax_scale(imagem_gray, feature_range=(0,255))
    imagem_gray /= (imagem_gray.max()/1.0)
    return imagem_gray


def segmentaMelanomaComBorda(imagem):
    # Converte para cinza
    im = imagem
    #im = imagem[:,:,1]
    # Borra a imagem para ajudar a segmentar os pelos
    img_med = median(im,np.ones((100,100)))
    # pega o melhor limiar de otsu
    limiar = threshold_otsu(im)
    # Remove pedaços soltos
    img_bin = closing(im > limiar, square(5))
    # Inverter preto e branco para poder remover as bordas
    inverted_img = util.invert(img_bin)
    # Remove a borda
    cleared = clear_border(inverted_img)
    
    # Condição
    if cleared.mean() < 0.1:
        cleared = inverted_img
        
    #le os labels da mascara
    labels  =  label ( cleared ) 

    #pega as regioes
    regions = regionprops(labels)
    
    try:
        #seleciona apenas a maior regiao
        maior_region = reduce((lambda x, y: x if x['Area']>y['Area'] else y), regions)
    except:
        #print('preta')
        return cleared

    #gera a a mascara da maior regiao
    mask = np.zeros(labels.shape)
    mask[maior_region['coords'][:,0],maior_region['coords'][:,1]]=1

    return mask

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
    
    
########################## usando joblib
def ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico):
    resultado = joblib.Parallel(n_jobs=NUMERO_JOBS)(joblib.delayed(processa_rgb_segmenta_dice)(x,y,z,imagens[nr],mascaras_medico[nr]) for nr in range(len(imagens)))
          
    return np.array(resultado).mean()
#######################################################################



# variavel para controlar o print de qual individuo começou a retomada do processo
retomada = False

# Função de avaliação dos individuos (fitness evaluation)
def Avaliacao(populacao,geracao,imagens,mascaras_medico,n_genes):
    
    global retomada
    
    populacao_nova = np.zeros((populacao.shape[0],n_genes+1))
    
    # Se for a população inicial, ai nao coloca dice, se for qualquer outra geração, coloca o dice.
    
    if populacao.shape[1] == 3:
        for i,p in enumerate(populacao):
            x = p[0]
            y = p[1]
            z = p[2]
            
            populacao_nova[:,:-1] = populacao
            ########print('Inicio do Individuo ',i,' - X ',x,' Y ',y,' Z ',z)
            dice = ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico)
            print('##### Fim do Individuo ',i,' - X ',x,' Y ',y,' Z ',z,' - Dice: ',dice)
            populacao_nova[i,-1] = dice
            # Salvando o dice de cada individuo que rodou na rede
            np.savetxt(PASTA+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
            ########print('individuo salvo')
    
    # Quando nao for a populacao inicial e sim a de um csv lido
    else:
        
        # Verificar se todos os individuos ja foram avaliados
        # todos avaliados
        if populacao.shape[0] == np.count_nonzero(populacao[:,-1]):
            populacao_nova[:,:-1] = populacao[:,:-1]
            # geracao += 1 # comentei pq ja incrementa na função geral
            for i,p in enumerate(populacao):
                x = p[0]
                y = p[1]
                z = p[2]

                ########print('Inicio do Individuo ',i,' - X ',x,' Y ',y,' Z ',z)
                dice = ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico)
                print('##### Fim do Individuo ',i,' - X ',x,' Y ',y,' Z ',z,' - Dice: ',dice)
                populacao_nova[i,-1] = dice
                # Salvando o dice de cada individuo que rodou na rede
                np.savetxt(PASTA+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
                ######## print('individuo salvo')
        
        # nem todos foram avaliados ou nenhum
        else:
            populacao_nova = populacao
            for i,p in enumerate(populacao):
                x = p[0]
                y = p[1]
                z = p[2]
                
                # condição para calcular apenas dos que tem dice 0.0
                if populacao_nova[i,-1] == 0.0:
                    # Condição para mostrar em qual individuo retomou
                    if retomada == True:
                        print("##################  Retomando processo do individuo ", i)
                        retomada = False
                        
                    #######print('Inicio do Individuo ',i,' - X ',x,' Y ',y,' Z ',z)
                    dice = ComBordaOuSemBorda(x,y,z,imagens,mascaras_medico)
                    print('##### Fim do Individuo ',i,' - X ',x,' Y ',y,' Z ',z,' - Dice: ',dice)
                    populacao_nova[i,-1] = dice

                    # Salvando o dice de cada individuo que rodou na rede
                    np.savetxt(PASTA+'/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
                    #########print('individuo salvo')
                    retomada = False
 
    # ordenar pelo fitness
    populacao_nova = populacao_nova[populacao_nova[:,3].argsort()[::-1]]
    
    # Salvando a população nova com todos os dices agora ordenada
    np.savetxt('populacao60mutacao/dados_evolucao-geracao'+str(geracao)+'.csv',populacao_nova,delimiter=",",fmt='%s')
    print("população ordenada salva")
    
    return populacao_nova, populacao_nova[:,3].mean()


############ Seleção de pai e mãe para cruzamento

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



########################### Cruzamento


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



########################### # Mutação


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


##################### NOVA POPULAÇÃO 

def novaPopulacao(population, childrens, elitism=20):
    
    nova_populacao = np.zeros((population.shape[0],population.shape[1],population.shape[2])).astype(int)
    
    tamanho_populacao = population.shape[0]
    # quantidade de filhos gerados e de pais que seguirão por elitismo para a nova população
    qtd_elitism = tamanho_populacao*elitism//100
    qtd_filhos = tamanho_populacao - qtd_elitism
    
    nova_populacao[:qtd_elitism] = population[:qtd_elitism]
    nova_populacao[qtd_elitism:] = childrens[:qtd_filhos]
    
    return nova_populacao


#########################################3 # função central

def evoluirOtsu(imagens,mascaras_medico,n_indiduals,n_genes,decimal_places,begin_gene,end_gene,m_probability,begin_allele,end_allele,elitism):
    global retomada
    print('######## inicio da evolução ########')
    dados_geracao = []
    
    #################### aqui quero gerar binaria e converter pra real
    # inicialização da nova população
    
    
    #inicio binario e converto pra real
    population_init = binary_initialization(population=n_indiduals, parameters=n_genes, n_bits=14)
    population_init = populacao_binaria_to_real(population_init,first=True)

    ####################
    
    # Agora vamos ver se tem alguma geração com individuos, se não tiver ai que usa a população inicial
    csv_dados_evolucao = glob.glob(PASTA+'/dados_evolucao-geracao*.csv')
    csv_dados_evolucao.sort()
    if len(csv_dados_evolucao) == 0:
        retomada = False
        print("################## Começando com a população inicial / geração 0")

        # Fitness
        geracao = 0
        #################### aqui a população ja entra como real
        populacao, aptidao = Avaliacao(population_init,geracao,imagens,mascaras_medico, n_genes)

        #np.savetxt('populacao/dados_evolucao-geracao'+str(geracao)+'.csv',populacao,delimiter=",",fmt='%s')

        print(" ######################################## ")
        print('Geração: ',geracao, ' / Media DSC: ', aptidao)
        print(" ######################################## ")

        dados_geracao.append(populacao)
    else:
        retomada = True
        #print("################## Retomando processo da geração", len(csv_dados_evolucao)-1)
        # pegar o ultimo csv
        population = np.genfromtxt(csv_dados_evolucao[-1],delimiter=',')
        geracao = len(csv_dados_evolucao)-1
        # se todos ja tiverem sido avalidado, incrementar a geração
        if population.shape[0] == np.count_nonzero(population[:,-1]):
            geracao +=1
        
        print("################## Retomando processo da geração", geracao)
        
        populacao, aptidao = Avaliacao(population,geracao,imagens,mascaras_medico, n_genes)
        print(" ######################################## ")
        print('Geração: ',geracao, ' / Media DSC: ', aptidao)
        print(" ######################################## ")
        
        dados_geracao.append(populacao)
        
    
    # condição de parada
    while aptidao <= 0.99 and geracao < NUMERO_GERACOES: ########## AQUI DEFINO QUANTAS GERAÇÕES VAI SER
        
        #################### aqui eu pego a populacao e converto pra populacao binaria
        ##### nao precisa ter o dice
        populacao_binaria = populacao_real_to_binaria(populacao)
        
        
        # Seleção de pais e mães
        pai, mae = tournament_selection(populacao_binaria)
        
        # Recombinação
        filhos = uniform_crossover(pai, mae)
        
        # mutando os filhos
        filhos_mutados = mutation_all(filhos)

        # Concepção da nova população
        nova_populacao_binaria = novaPopulacao(populacao_binaria, filhos_mutados, elitism=20)
        
        geracao += 1
        
        #################### aqui converto de volta pra real a nova população
        ##### aqui precisa ter o campo do dice, porem com 0
        nova_populacao = populacao_binaria_to_real(nova_populacao_binaria,first=False)
        
        ##### print(nova_populacao.shape)
        # Fitness novo
        populacao, aptidao = Avaliacao(nova_populacao,geracao,imagens,mascaras_medico,n_genes)
        
        print(" ######################################## ")
        print('Geração: ',geracao, ' / Media DSC: ', aptidao)
        #print(populacao)
        print(" ######################################## ")
        
        # populacao é toda a populacao de uma geracao
        dados_geracao.append(populacao)
        #np.savetxt('populacao/dados_evolucao-geracao'+str(geracao)+'.csv',populacao,delimiter=",",fmt='%s')
    
    print('######## fim da evolução ########')
    # dados_geracao são todas as geracoes
    return dados_geracao



############################### AQUI ONDE CHAMA A EXECUÇÃO ###########################
#######################################################################################

# so ta valendo o parametro n_individuals, que é o numero de individuos
dados_geracao = evoluirOtsu(x_train,y_train,n_indiduals=QTD_INDIVIDUOS,n_genes=3,decimal_places=4,begin_gene=0,end_gene=1,m_probability=5,begin_allele=0,end_allele=1,elitism=20)


#############################################################################################
#############################################################################################


def save_results(dados):
    # cria a nova matrix com a coluna da geração
    dadosnovo = np.zeros((dados.shape[0],dados.shape[1],dados.shape[2]+1))
    
    # nova matrix recebe os dados das gerações
    dadosnovo[:,:,:-1] = dados[:,:,:]
    
    # adiciona o numero das gerações
    for i in range(dadosnovo.shape[0]):
        for j in range(dadosnovo.shape[1]):
            dadosnovo[i,:,4] = i
    
    # criação do dataframe que ira receber os resultados com as gerações
    nova_populacaogigante = df = pd.DataFrame(columns=['X', 'Y', 'Z', 'ACC', 'Geration'])
    
    for i in range(dadosnovo.shape[0]):
        nova_populacao = pd.DataFrame(columns=['X', 'Y', 'Z', 'ACC', 'Geration'],data=dadosnovo[i])
        nova_populacaogigante = nova_populacaogigante.append(nova_populacao, ignore_index=True)
    
    return nova_populacaogigante




# FUSION OF COLOR BANDS USING GENETIC ALGORITHM TO SEGMENT MELANOMA

Melanoma is often associated with changes in the color,size or shape of a mole.  Several tools, including smartphoneapps,  have  been  developed  for  the  detection  of  melanomathrough medical images. To interpret the information in theseimages efficiently, it is necessary to isolate a region of inter-est.  In this research, we analyze the impact of evolutionarycomputing on the segmentation of melanoma through the fu-sion of color bands in the images.   The tests performed onthe PH2 image base showed a 20% improvement in the aver-age Dice compared to the standard intensity, showing that the algorithm is promising.

[Acessar o artigo completo](https://ieeexplore.ieee.org/document/9153438)

## Instructions for using the algorithm

#### Create a virtual environment and install the dependencies

    pip install -r requeriments.txt

#### In the file evolve_otsu_binario.py go to SETTINGS and set the default values for:    
    - FOLDER_RESULTS     
    - NUMBER_INDIVIDUALS     
    - NUMBER_GENERATIONS            
    - NUMBER_JOBS      

#### Run the evolve_otsu_binario.py file
    
    python evolve_otsu_binario.py

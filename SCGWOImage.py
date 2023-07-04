# python implementation of SCE with GWO to blend two images

import random
import copy  # array-copying convenience
import cv2 as cv
import numpy as np
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim

class agent:
    def __init__(self, positionx,positiony, fitness):
        self.positionx = positionx
        self.positiony = positiony
        self.fitness = fitness


def GWO(population,L,U,NoIter):
    for iter in range(1, NoIter + 1):
        a = 2 * (1 - (iter / (NoIter + 1)))
        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
        wolfs=[alpha_wolf, beta_wolf, gamma_wolf]
        for i in range (len(population)):
            Xsum=0
            Ysum = 0
            for j in range(3):
                r1=random.random()
                r2=random.random()
                A=2*a*r1-a
                C=2*r2
                Dx=abs(C*wolfs[j].positionx-population[i].positionx)
                Dy=abs(C*wolfs[j].positiony-population[i].positiony)
                Xsum+=wolfs[j].positionx-A*Dx
                Ysum+=wolfs[j].positiony-A*Dy
            Xmean=Xsum/3
            Ymean=Ysum/3
            dst = cv.addWeighted(src1, Xmean, src2, Ymean, 0.0)
            uiqi1_1 = ssim(src1, dst)
            uiqi1_2 = ssim(src2, dst)
            NewFit = (uiqi1_1 + uiqi1_2) / 2

            if NewFit > population[i].fitness and Xmean>L and Xmean<U and Ymean>L and Ymean<U:
                population[i].positionx = Xmean
                population[i].positiony = Ymean
                population[i].fitness = NewFit

    population = sorted(population, key=lambda temp: temp.fitness,reverse=True)
    return(population)



# -------Shuffled complex function---------
def shuffled(population,m, n, L, U,it2):
    newpopulation=list()
    remain=list()


    memep = list()
    while m >=1:

        population = sorted(population, key=lambda temp: temp.fitness,reverse=True)


        for i in range(m):

            memep.clear()

            for j in range(n) :


                memep.append(population[i + j*m])





            memep=GWO(memep,L,U,it2)

            for k in range(3):

                newpopulation.append(memep[k])

            for k in range(3,n):
                remain.append(memep[k])

        if len(newpopulation)%n!=0:


            if len(newpopulation)<=3:
                m=0

            if len(newpopulation)<n and len(newpopulation)>3:
                m=1
                n=len(newpopulation)

            if len(newpopulation)>n:
                remain=sorted(remain, key=lambda temp:temp.fitness,reverse=True)
                if (len(newpopulation)+n-(len(newpopulation)%n))/n!=m:
                    for s in range(n-(len(newpopulation)%n)):
                        newpopulation.append(remain[s])


                m = int(len(newpopulation) / n)
        else:m = int(len(newpopulation) / n)


        population.clear()

        for o in range(len(newpopulation)):

            population.append(newpopulation[o])
        newpopulation.clear()
        remain.clear()


    return (population)

def shuffledmain(NoI,L,U,m,n):
    seedx = list()
    seedy = list()
    fit=list()
    for i in range(m*n):
        seedx.append(L + random.random() * (U - L))
        seedy.append(L + random.random() * (U - L))

        dst = cv.addWeighted(src1, seedx[i], src2, 1-seedx[i], 0.0)
        uiqi1_1 = ssim(src1, dst)
        uiqi1_2 = ssim(src2, dst)
        fit.append((uiqi1_1 + uiqi1_2) / 2)      # -------fitness function--------

    population = [agent(seedx[i],seedy[i], fit[i]) for i in range(len(seedx))]
    population = shuffled(population, m, n, L, U, NoI)
    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
    return population

def OPmain(img1,img2):
    global src1
    global src2
    src1 = img1
    src2=img2
    NoI = int(input("Enter the number of Iterations"))
    L = int(input("Min="))
    U = int(input("Max="))


    m = int(input("Number of Complex="))
    n = int(input("Number of wolves in the pack length="))



    population=shuffledmain(NoI,L,U,m,n)
    Best = max(population, key=lambda temp: temp.fitness)
    alpa=Best.positionx
    beta=Best.positiony
    print(alpa,beta)
    print('ssim',Best.fitness)
    dst = cv.addWeighted(src1, alpa, src2, 1-alpa, 0.0)

    src1 = np.ravel(src1)
    src2 = np.ravel(src2)
    dst=np.ravel(dst)
    mi = mutual_info_score(src1, dst)
    mi1 = mutual_info_score(src2, dst)
    mi0=(mi+mi1)/2
    print('mi=',mi0)
    # [display]
    cv.imshow('dst', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return [alpa, beta]






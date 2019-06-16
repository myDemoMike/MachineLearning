# /usr/bin/env python


import math

TrainData = []

LMD = 0.5  # lambda, step size

MinLMD = 1.0e-30
ParamNum = 5  # number of parameters
Betas = [0.0, 0.0, 0.0, 0.0, 0.0]
NewBetas = [0.5, 0.5, 0.5, 0.5, 0.5]


# read training data
def ReadData(trainingDataFile):
    infile = open(trainingDataFile, 'r')
    sline = infile.readline().strip()
    while len(sline) > 0:
        fields = sline.strip().split(" ")

        fvector = []

        for field in fields:
            fvector.append(float(field))

        # append fake feature for beta0
        fvector.append(1.0)

        # used for computing exp(Z) (Z is the dot product of feature weight vector and example vector)
        fvector.append(0.0)

        # print vector

        TrainData.append(fvector)
        sline = infile.readline().strip()
    infile.close()
    # for eachrow in TrainData:

    #	for eachfield in eachrow:

    #		print eachfield,
    #	print
    print(len(TrainData), "lines loaded!")


# compute exp(Z)(Z is the dot product of feature weight vector and example vector)
def ExpZ(fv, weis):
    f = 0.0

    for i in range(len(weis)):
        f += weis[i] * fv[i]

    return math.exp(f)


# compute the l2-norm for vector v
def Mode(v):
    sum = 0.0

    for f in v:
        sum += f * f

    sum = math.sqrt(sum)

    return sum


# compute the negative log-likelihood of the whole training data
def ComputeLL(weis):
    ll = 0.0

    f = 0.0

    for anexample in TrainData:
        f = ExpZ(anexample[2:], weis)

        ll += anexample[0] * math.log(f / (float)(1 + f))

        ll -= anexample[1] * math.log(1 + f)

    return -ll


# Gradient Descent
def Iterate():
    global LMD
    f = 0.0

    ll = 0.0

    i = 0

    for anexample in TrainData:
        f = ExpZ(anexample[2:], Betas)

        TrainData[i][7] = f

        i += 1

        ll += anexample[0] * math.log(f / (float)(1 + f))

        ll -= anexample[1] * math.log(1 + f)

    ll = -ll

    # gradient calculation
    wv = []

    for i in range(ParamNum):

        sum = 0.0

        for anexample in TrainData:
            f = anexample[7]
            newf = anexample[1] * f - anexample[0]

            newf *= 1 / (float)(1 + f)

            sum += newf * anexample[i + 2]

        wv.append(sum)

    mode = Mode(wv)

    # line search
    newll = ll

    LMD = 2 * LMD

    while ll <= newll and LMD > MinLMD:

        LMD /= 2

        # print LMD

        for i in range(0, ParamNum):
            NewBetas[i] = Betas[i] - LMD * wv[i] / mode

        newll = ComputeLL(NewBetas)

    for i in range(0, ParamNum):
        Betas[i] = NewBetas[i]

    print(ll)
    print(Betas)


if __name__ == '__main__':
    trainingFile = ".\\data\\LRTrainNew.txt"
    ReadData(trainingFile)
    itnum = 0

    while itnum < 500 and LMD > MinLMD:
        itnum += 1

        print(itnum)

        Iterate()



# 356
# 1633.745217
# [0.5825393278683701, 0.7144661903649068, -0.06573471376316616, 0.4934347514500628, -6.157805832121684]

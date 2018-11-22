import numpy as np
import theano
import theano.tensor as T
from pegpy.peg import *
import os

# Numpyの設定
np.random.seed(100)
upper = -.002
lower = 0.002
samples = 100

#################################
# configure hyperparameter
margin = 1.0
learnRate = .003
momentum = 0.1

decay = momentum/(1-momentum)
alpha = learnRate * (1-momentum)

C = .1/17340.0
C_2 = C/2

feature_dimension = 30

lam = 0.004 # 正則化項のl(Hyper Parameter)

################################
 ## Treeの内部表現
class VectorTree(object):
    __slots__ = [
        "numberOfSiblings"
        , "positionInSiblings"
        , "tag"
        , "child"
    ]

    def __init__(self):
        self.numberOfSiblings = 0
        self.positionInSiblings = 0
        self.tag = None
        self.child = []

    def __len__(self):
        return len(self.child)

    def serialize(self): # 全てのノードを取り出す
        nodelist = [self]
        for childnode in self.child:
            nodelist.extend(childnode.serialize())
        return nodelist

    def leafs(self):
        return list(filter(lambda item: len(item) == 0, self.serialize()))

    def numberOfLeaf(self):
        return len(self.leafs())

# Vector treeとなっているが実際はstringのタグが入っている
def parse2vector(parsetree, sibpos = 0, sibnum = 0):
    top = VectorTree()
    top.tag = parsetree.tag
    top.numberOfSiblings = sibnum
    top.positionInSiblings = sibpos
    sibNum = len(parsetree)
    if sibNum != 0:
        sibPosCounter = 0
        for label, subtree in parsetree:
            top.child.append(parse2vector(subtree, sibpos = sibPosCounter, sibnum = sibNum-1))
            sibPosCounter += 1
    return top

# Parser
grammar = Grammar("x")
grammar.load(os.path.dirname(os.path.abspath(__file__)) + "/test_grammar/math.tpeg")
parser = nez(grammar)
# tagのマップを作る
tagList = ["Infix", " ", "Int","Plus","Mul"]#grammar.tagAll()
initTagMap = {}
for tag in tagList:
    initTagMap[tag] = np.random.rand(feature_dimension)
tagMap = [initTagMap, initTagMap]

## Training Data
fin = open(os.path.dirname(os.path.abspath(__file__)) + "/expressions.txt")
progs = fin.readlines()
fin.close()
parsetrees = [parser(prog) for prog in progs]
trainData = np.array([parse2vector(parsetree) for parsetree in parsetrees])
#################################
# configure network
Wleft = theano.shared(np.random.uniform(low=lower,high=upper,size=(feature_dimension,feature_dimension)),name = "wleft", borrow=True)
Wright = theano.shared(np.random.uniform(low=lower,high=upper,size=(feature_dimension,feature_dimension)), name = "wright", borrow=True)
biase = theano.shared(np.random.uniform(low=lower,high=upper,size=feature_dimension), name = "biase", borrow = True)

def mulLandW(node, nodevector):
    n = node.numberOfSiblings
    l = node.numberOfLeaf()
    if n == 0:# Unary tree だった場合
        Weight = 0.5 * l * Wleft + 0.5 * l * Wright # simbol expression
        return T.dot(Weight, nodevector)
    else:
        i = float(node.positionInSiblings)
        eta_l = (n - i) / n
        eta_r = i / n
        Weight = eta_l * l * Wleft + eta_r * l * Wright# simbol expression
        return T.dot(Weight, nodevector)

# error function for a vector tree
def j(vecTree):
    if len(vecTree) == 0:
        return 0.0
    else:
        # for positive
        currentVector = tagMap[0][vecTree.tag]
        numOLeafs = vecTree.numberOfLeaf()
        mlw = [mulLandW(ch, tagMap[0][ch.tag]) for ch in vecTree.child] # simbol expression
        nextVector = T.tanh((1.0/numOLeafs) * T.sum(mlw) + biase)
        '''
        Eval NextVector w.r.t. the current model parameters
        and update a new code for tagMap[1]
        '''
        evalNextVector = theano.function([], nextVector)
        tagMap[1][vecTree.tag] = evalNextVector()
        d_positive = (currentVector - nextVector).norm(L=2)
        '''
        Done
        '''
        #for negative
        negNodeIndex = np.random.randint(0,len(vecTree))
        if negNodeIndex == len(vecTree):
            d_negative = (np.random.rand(feature_dimension) - nextVector).norm(L=2)
        else:
            child_neg = vecTree.child
            child_neg[negNodeIndex].tag = "negative" # negative sample を示すタグに書き換え
            tagMap[0]["negative"] = np.random.uniform(low=lower,high=upper,size=feature_dimension)# negative sampleの値を更新
            neglw = [mulLandW(ch, tagMap[0][ch.tag]) for ch in child_neg] # simbol expresison
            nextVector_neg = T.tanh((1 / numOLeafs ) * T.sum(neglw) + biase)
            d_negative = (currentVector - nextVector_neg).norm(L=2)
        er = margin + d_positive - d_negative
        return T.max([0.0,er])


def objective(vectree):
    treeList = vectree.serialize()
    serr = [j(tree) for tree in treeList]# simbol expression
    totalError = T.sum(serr) * 0.5 / len(treeList)
    regularizer = (lam * 0.5 * ( Wleft.norm(L=2) + Wright.norm(L=2) )) / pow(feature_dimension, 2)
    return totalError + regularizer


TRAIN_EPOCH = 1
for epoch in range(TRAIN_EPOCH):
    np.random.shuffle(trainData)
    current_cost = 0.0
    for datum in trainData:
        djdWleft = T.grad(cost = theano.gradient.grad_clip(objective(datum), -1, 1) , wrt = Wleft)
        print('left')
        print(djdWleft.eval())
        djdWright = T.grad(cost = theano.gradient.grad_clip(objective(datum), -1, 1) , wrt = Wright)
        print('right')
        print(djdWright.eval())
        djdbiase = T.grad(cost = theano.gradient.grad_clip(objective(datum), -1, 1) , wrt = biase)
        print('biase')
        print(djdbiase.eval())
        #djdtheta = T.grad(cost=objective(datum),wrt=theta)
        updates = [(Wleft, Wleft - learnRate * djdWleft), (Wright, Wright - learnRate * djdWright), (biase, biase - learnRate * djdbiase)]
        train_model = theano.function(inputs=[],outputs=objective(datum), updates=updates)
        current_cost = train_model()
        print(str(current_cost))
    tagMap[0] = tagMap[1] ## tagmapの更新
    print(str(epoch) + str(current_cost))
import numpy as np
import theano
import theano.tensor as T
from pegpy.peg import *
import os

theano.config.optimizer='fast_compile'

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
        , "isnegativesample"
    ]

    def __init__(self):
        self.numberOfSiblings = 0
        self.positionInSiblings = 0
        self.tag = None
        self.child = []
        self.isnegativesample = False

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
tag2idx = {"Infix" : np.array([0,0,0,0,1]), " " : np.array([0,0,0,1,0]), "Int" : np.array([0,0,1,0,0]),"Plus" : np.array([0,1,0,0,0]),"Mul" : np.array([1,0,0,0,0])}
tag2vec = theano.shared(np.random.rand(feature_dimension,len(tag2idx)),name='tag2vec', borrow = True)
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

def getVec(tagname):
    return T.dot(tag2vec, tag2idx[tagname])

def mulLandW(node):
    n = node.numberOfSiblings
    l = node.numberOfLeaf()
    
    def ret(nodevector):
        if n == 0:# Unary tree だった場合
            Weight = 0.5 * l * Wleft + 0.5 * l * Wright # simbol expression
            return T.dot(Weight, nodevector)
        else:
            i = float(node.positionInSiblings)
            eta_l = (n - i) / n
            eta_r = i / n
            Weight = eta_l * l * Wleft + eta_r * l * Wright# simbol expression
            return T.dot(Weight, nodevector)

    if node.isnegativesample:
        return ret(theano.shared(np.random.uniform(low=lower,high=upper,size=(feature_dimension))))
    else:
        return ret(getVec(node.tag))

# error function for a vector tree
def j(vecTree):
    if len(vecTree) == 0:
        return 0.0
    else:
        # for positive
        currentVector = getVec(vecTree.tag)
        numOLeafs = vecTree.numberOfLeaf()
        mlw = [mulLandW(ch) for ch in vecTree.child] # simbol expression
        nextVector = T.tanh((1.0/numOLeafs) * T.sum(mlw) + biase)
        d_positive = (currentVector - nextVector).norm(L=2)
        #for negative
        negNodeIndex = np.random.randint(0,len(vecTree))
        if negNodeIndex == len(vecTree):
            d_negative = (np.random.rand(feature_dimension) - nextVector).norm(L=2)
        else:
            vecTree.child[negNodeIndex].isnegativesample = True ##あとでFasleに戻すことを忘れない
            neglw = [mulLandW(ch) for ch in vecTree.child] # simbol expresison
            vecTree.child[negNodeIndex].isnegativesample = False ## 戻した
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
    samplecount = 0
    for datum in trainData:
        djdWleft = T.grad(cost = objective(datum), wrt = Wleft)
        print('left')
        print(djdWleft.eval())
        djdWright = T.grad(cost = objective(datum), wrt = Wright)
        print('right')
        print(djdWright.eval())
        djdbiase = T.grad(cost = objective(datum), wrt = biase)
        print('biase')
        print(djdbiase.eval())
        djdtag2vec = T.grad(cost= objective(datum), wrt= tag2vec)
        print('tag2vec')
        print(djdtag2vec.eval())
        updates = [(Wleft, Wleft - learnRate * djdWleft)
                 , (Wright, Wright - learnRate * djdWright)
                 , (biase, biase - learnRate * djdbiase)
                 , (tag2vec, tag2vec - learnRate * djdtag2vec)]
        train_model = theano.function(inputs=[],outputs=objective(datum), updates=updates)
        current_cost = train_model()
        print('samplecount:' + str(samplecount) + 'current_coust:' + str(current_cost))
    print('epoch:' + str(epoch) + 'samlecount:' + str(samplecount) +'current_cost:' + str(current_cost))
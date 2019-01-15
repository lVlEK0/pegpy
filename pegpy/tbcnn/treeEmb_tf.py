import numpy as np
from pegpy.peg import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
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

    def numberOfLeafs(self):
        return len(self.leafs())

# Parser
grammar = Grammar("x")
grammar.load(os.path.dirname(os.path.abspath(__file__)) + "/test_grammar/math.tpeg")
parser = nez(grammar)

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

# tagのマップを作る
tag2idx = {"Infix" : tf.constant([0,0,0,0,1]), " " : tf.constant([0,0,0,1,0]), "Int" : tf.constant([0,0,1,0,0]),"Plus" : tf.constant([0,1,0,0,0]),"Mul" : tf.constant([1,0,0,0,0])}
tag2vec = tf.Variable(tf.random_uniform([feature_dimension,len(tag2idx)],lower,upper))

fin = open(os.path.dirname(os.path.abspath(__file__)) + "/expressions.txt")
progs = fin.readlines()
fin.close()
parsetrees = [parser(prog) for prog in progs]
vectortrees = [parse2vector(parsetree) for parsetree in parsetrees]
#trX, testX = train_test_split(np.array(vectortrees), train_size=0.8)
trX = np.array(vectortrees)
#################################
# configure network 
Wleft = tf.Variable(tf.random_uniform([feature_dimension,feature_dimension], lower, upper))
Wright = tf.Variable(tf.random_uniform([feature_dimension,feature_dimension],lower, upper))
biase = tf.Variable(tf.random_uniform([feature_dimension], lower, upper))

def matvecmul(mat, vec):
    return tf.matmul(mat,tf.expand_dims(vec,-1))

def getVec(tagname):
    val = matvecmul(tag2vec, tag2idx[tagname])
    return tf.squeeze(val)

def mulLandW(node):
    n = node.numberOfSiblings
    l = node.numberOfLeafs()

    def ret(nodevector):
        if n == 0:# Unary tree だった場合
            Weight = tf.scalar_mul(0.5 * l,Wleft) + tf.scalar_mul(0.5 * l,Wright)
            return matvecmul(Weight, nodevector)
        else:
            i = float(node.positionInSiblings)
            eta_l = (n - i) / n
            eta_r = i / n
            Weight = tf.scalar_mul(eta_l * l,Wleft) + tf.scalar_mul(eta_r * l, Wright)
            return matvecmul(Weight, nodevector)
    
    if node.isnegativesample:
        return ret(tf.constant(tf.random_uniform([feature_dimension],lower, upper)))
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
        mlw = [mulLandW(ch) for ch in vecTree.child]
        nextVector = tf.nn.tanh((1.0/numOLeafs) * sum(mlw) + tf.expand_dims(biase,-1))
        d_positive = tf.norm(currentVector - nextVector)
        #for negative
        negNodeIndex = np.random.randint(0, len(vecTree))
        if negNodeIndex == len(vecTree):
            d_negative = tf.norm(tf.random_uniform([feature_dimension],lower, upper) - nextVector)
        else:
            vecTree.child[negNodeIndex].isnegativesample = True
            neglw = [mulLandW(ch) for ch in vecTree.child]
            vecTree.child[negNodeIndex].isnegativesample = False
            nextVector_neg = tf.nn.tanh(tf.scalar_mul(1/numOLeafs, sum(neglw)) + tf.expand_dims(biase,-1))
            d_negative = tf.norm(currentVector - tf.squeeze(nextVector_neg))
        er = margin + d_positive - d_negative
        return tf.math.maximum(0.0,er)

def objective(vectree_idx):
    vectree = trX[vectree_idx]
    treeList = vectree.serialize()
    totalError = sum([j(tree) for tree in treeList]) * 0.5 / len(treeList)
    regularizer = (lam * 0.5 * (tf.norm(Wleft) + tf.norm(Wright) )) / pow(feature_dimension, 2)
    return totalError + regularizer

### Model Parameters
X = tf.placeholder(tf.float32)
train_step = tf.train.MomentumOptimizer(learnRate,momentum).minimize(objective(X))

### Session
#BATCH_SIZE = 20
TRAIN_EPOCH = 10

with tf.Session() as sess:
    tf.global_variables_initializer()
    idxarray = np.array(range(len(trX)))
    # training SGD
    for epoch in range(TRAIN_EPOCH):
        np.random.shuffle(idxarray)
        for idx in idxarray:
            sess.run(train_step, feed_dict={X:idx})
            print(idx)
        #minibatch
        # p = np.random.permutation(range(len(trX)))
        # trX = trX[p]
        # for start in range(0, len(trX), BATCH_SIZE):
        #     end = start + BATCH_SIZE
        #     sess.run(momentumOptimizer(trX))
        # print training error
        #print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))
    # varidational error
    #print(output)

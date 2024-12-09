import numpy as np

# aggregation of change vectors for GNIMBUS

# change vectors are encoded as vectors of integer numbers between 0 and 2
# 0: objective can be worsened
# 1 objective can stay the same
# 2: objective can be improved
# they are indexed by interpreting them as base 3 numbers + 1

# these two functions translate between change vectors and their index numbers
# both work for several change vectors at once
"""
workDom<-function(i,j,nObj) {
# work function for creating dominance relation
# to be called via outer

# not that change vectors are all different, so we do not have to care about equal vectors
dom<-apply(deCode(i,nObj) >= deCode(j,nObj),1,all)
dom}

testDom<-function(nObj) {
# this simply calls workDom in an outer call
relDom<-outer(1:(3^nObj),1:(3^nObj),workDom,nObj=nObj)
diag(relDom)<-FALSE
relDom}
"""
# find a change vector from its number. Numbers are 1 plus the base 3 coding
def decode(x, n_objs):
    x = -x-1
    pass

def encode(v, n_objs):
    pass

def work_dom(i, j, n_objs):
    pass


def test_dom(n_objs):

    def test_ufunc(a, b):
        return a + b

    test_arr = np.frompyfunc(test_ufunc, 2, 1)
    print(test_arr(np.array(2, 1)))
    return test_arr
    # rel_dom = np.outer()

def work_swap(i, j, a, b, ref):
    pass


def test_swap(ref):
    pass

def test_k_ratio(kr, n_objs):
    pass

def make_ranks(rel):
    pass

def nd_compromise(comp, ranks):
    pass


# main funciton to find compromise change vector
# inputs:
# refs: a matrix containing individual change vectors as rows
# kr: a threshold for the ratio test (optional)
# printINtermediate: binary variable whetehr intermediate results should be printed
# output: a list wiht two elements
# ranks: a matrix contianing the ranks of all changes vectors for eah member (mebers are on the columns)
# compromise: one or several compromise (maxmin) change vactors
if __name__ == "__main__":

    # a matrix containing individual change vectors as rows. Then, objectives are the columns.
    # 3 DMs with 4 objective problem
    # At this point, we must have already checked for validity (each member is worsening something if improving something)
    refs = np.array([
        [2, 0, 1, 2],
        [1, 0, 2, 1],
        [2, 1, 0, 1]
    ])

    print("at main")
    n_dms, n_objs = refs.shape

    # this is how I can create the work_dom as ufunc to use in the outer
    def test_ufunc(a, b):
        print("at ufunc")
        return a + b
    """
    In [3]: a = np.arange(5)
In [4]: b = np.arange(8)[::-1]
In [5]: np.subtract.outer(a,b)
Out[5]:
array([[-7, -6, -5, -4, -3, -2, -1,  0],
       [-6, -5, -4, -3, -2, -1,  0,  1],
       [-5, -4, -3, -2, -1,  0,  1,  2],
       [-4, -3, -2, -1,  0,  1,  2,  3],
       [-3, -2, -1,  0,  1,  2,  3,  4]])
    """

    test_arr = np.frompyfunc(test_ufunc, 2, 1)
    print(test_arr(np.array([2, 10]), np.array([1, 0])))
    # dominance relation
    # rel = test_dom(n_objs)
    # print(rel(np.array(2, 1)))

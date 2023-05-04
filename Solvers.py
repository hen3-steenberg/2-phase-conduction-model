import scipy.sparse.linalg
import scipy.sparse as sp
def DirectSolve(shape, Cself, Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom, Residual):
    def get_index(i, j, k):
        return i + shape[0] * (j +  k * shape[1])
    
    Volume = shape[0] * shape[1] * shape[2]
    System = sp.lil_matrix((Volume, Volume))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                index = get_index(i, j, k)
                System[index, index] = Cself
                if i < shape[0] - 1 : 
                    eastindex = get_index(i + 1, j, k)
                    System[index,eastindex] = Ceast[i, j, k]
                if i > 0 :
                    westindex = get_index(i - 1, j, k)
                    System[index,westindex] = Cwest[i, j, k]
                if j < shape[1] - 1:
                    northindex = get_index(i, j + 1, k)
                    System[index, northindex] = Cnorth[i, j, k]
                if j > 0:
                    southindex = get_index(i, j - 1, k)
                    System[index, southindex] = Csouth[i, j, k]
                if k < shape[2] - 1:
                    topindex = get_index(i, j, k + 1)
                    System[index, topindex] = Ctop[i, j, k]
                if k > 0:
                    bottomindex = get_index(i, j, k - 1)
                    System[index, bottomindex] = Cbottom[i, j, k]
    System = System.tocsr()
    return sp.linalg.spsolve(System,Residual)
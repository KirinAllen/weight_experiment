import numpy as np

# Lemma Two matrices A (nxn) and B (nxn) are similar if and only if the rank of (lamdaI-A)^p equals the rank of (lamdaI-B)^p for any complex number lamda and for
# any integer p ,1 <= p <= n
def matrixs_similarity(A, B, eigvul1, eigvul2):
    I = np.mat(np.identity(A.shape[-1]))
    rankA = []
    rankB = []
    if A.shape[-1] == B.shape[-1]:
        for lam1 in eigvul1:
            rA = lam1*I-A
            rankA.append(np.linalg.matrix_rank(rA))
            for pa in range(A.shape[-1]):
                #print(np.linalg.matrix_rank((lam1*I-A)**p))
                rA = rA*(lam1*I-A)
                if pa == 5:
                    print(np.linalg.matrix_rank(rA),end = ' ')    
                rankA.append(np.linalg.matrix_rank(rA))
        for lam2 in eigvul2:
            rB = lam2*I-B
            rankB.append(np.linalg.matrix_rank(rB))
            for pb in range(B.shape[-1]):
                #print(np.linalg.matrix_rank((lam1*I-B)**p))
                rB = rB*(lam2*I-B)
                if pb == 5:
                    print(np.linalg.matrix_rank(rB), end=' ')
                rankB.append(np.linalg.matrix_rank(rB))
    
    if rankA == rankB:
        print('\n A is similar to B')
    else:
        print('\n A is not similar to B')


# 类比奇异值分解，定义一个非方阵到方阵的映射过程,并返回其变成方阵后的特征值和特征向量
def NoneSquareToSquareMatrix(A,B):
    A = A.dot(A.T)
    B = B.dot(B.T)

    print(A.shape)
    print(B.shape)

    A_eigenvalues, A_eigenvectors = np.linalg.eigh(A)
    B_eigenvalues, B_eigenvectros = np.linalg.eigh(B)

    return A, B, A_eigenvalues, B_eigenvalues


if __name__ == "__main__":

    A1 = np.matrix(np.array([[0,1,0],[0,0,1]]))
    B1 = np.matrix(np.array([[1,0,0],[0,1,0]]))

    print(A1.shape)
    print(B1.shape)

    A, B, A_eigenvalues, B_eigenvalues = NoneSquareToSquareMatrix(A1,B1)

    matrixs_similarity(A, B, A_eigenvalues, B_eigenvalues)

    print(' ')

    




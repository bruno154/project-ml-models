import numpy as np

class Knn:

    def __init__(self, k, dist):
        self._k = k
        self._dist = dist
    
    def dist_euclid(self,X1, X2):
        e = np.sqrt(np.power(X1-X2,2).sum(axis=1))
        return e

    def minkowsky(self, X1, X2, r=0.5):
        m = np.power(np.abs(np.power(X1-X2,r)).sum(axis=1),1/r)
        return m

    def fit(self,X_treino, y_treino, X_teste):
        x_treino = np.asarray(X_treino)
        y_treino = np.asarray(y_treino)
        x_teste = np.asarray(X_teste)

        shape_x_treino = x_treino.shape
        #shape_y_treino = y_treino.shape
        shape_x_teste = x_teste.shape

        labels = np.zeros([shape_x_treino[0],shape_x_treino[1]])

        for i in range(shape_x_teste[0]):
            rpt_test = np.tile(x_teste[i,:], (shape_x_treino[0],1))

            if self._dist == 'E':
                distc = self.dist_euclid(rpt_test,x_treino)
            elif self._dist == 'C':
                distc = self.minkowsky(rpt_test,x_treino)
            else:
                raise ValueError("Deu ruim irmão!!!")

            index_sort = np.argsort(distc)
            pos_labels = index_sort[:self._k]
            final_value = x_teste[pos_labels]
            labels[i] = final_value.sum(axis=0)

        return labels













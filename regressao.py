import numpy as np



class regressao_linear_simples:

    """

    Classe para o treinamento de modelos de Regressão Linear Simples 
    """

    def __init__(self):
        
        self._coef_ = 0
        self._intercept_ = 0
        
    
    def __str__(self):
        return "Classe para o treinamento de regressão linear simples "
    
    @property
    def intercept(self):
        return self._intercept_
    
    @property
    def coef(self):
        return self._coef_


    def fit(self,X,y):

        """

        :params X:  Dados de Treino 
        :params y:  variável target de treino
        :return: 
        """

        if len(X.shape) == 1:
            X.reshape(-1,1)

        # Variaveis auxiliares
        xmean = X.mean() 
        ymean = y.mean()

        # treinando os coefs
        beta = sum(((X-xmean)*(y-ymean)))/sum(((X-xmean)**2))
        alfa = ymean - (beta*xmean)

        #Coeficientes
        self._intercept_ = alfa
        self._coef_ = beta

        #return self._intercept_, self._coef_

    def predict(self, x):
        """
        :params  novos dados que o modelo não viu 
        :return  previsões do modelo
        """
        if len(x.shape) == 1:
            x.reshape(-1,1)

        self.predicted = self._intercept_ + np.dot(x,self._coef_)    
        return self.predicted

class regressao_linear_multipla:

    def __init__(self, fit_intercept=True):
        self._theta_ = 0
        self._fit_intercept = fit_intercept

    def __str__(self):
        return "Classe para o treinamento de regressão linear multipla"
    
    @property
    def theta(self):
        return self._theta_

    def fit_closed_form(self,X,y):
        if len(X.shape) == 1:
            X.reshape(-1,1)

        #Encontrando os coefs
        if self._fit_intercept:
            X_bias = np.c_[np.ones(X.shape), X]
        else:
            X_bias = X

        theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

        #atribuindo a matriz theta
        self._theta_ = theta
    
    def fit_batch_gd(self, X, y, eta=0.0000001 ,n_iter):
        """
        :params X:         Dados de treino variáveis independentes
        :params y:         Dados de treino variável  dependente
        :params eta:       Taxa de apredizagem
        :params n_iter:    numeros de interação
        """
        m = X.shape[1]
        if len(x.shape) == 1:
            x.reshape(-1,1)

        if self._fit_intercept:
            X_bias = np.c_[np.ones(X.shape), X]
        else:
            X_bias = X

        for i in range(n_iter):
            gradient = (2/m)*X_bias.T.dot((X_bias.dot(theta))-y)
            theta = theta - eta * gradient
        self._theta_ = theta

    
    def predict(self, x):
        """
        :params  novos dados que o modelo não viu 
        :return  previsões do modelo
        """

        if len(x.shape) == 1:
            x.reshape(-1,1)
        
        if self._fit_intercept:
            x_new = np.c_[np.ones(x.shape), x]
        else:
            x_new = x

        self.previsao_ = x_new.dot(self._theta_)
        
        return self.previsao_

class regressao_logistica_binaria:

    def __init__(self):
        self._w = 0
        self._b = 0
        self._n_variaveis = 0
        self._m_instancias = 0
        self._custo = 0
        self._custos = 0
        self._gradient = 0


    def init_pesos(self,X, b=0):
        n = X.shape[1]
        m = X.shape[0]
        w = np.zeros((1,n))
        b = b

        #gerando peso e bias
        self._w = w
        self._b = b
        self._n_variaveis = n
        self._m_instancias = m 

    def sigmoid(self,param_t):
        sig = 1/1+np.exp(-param_t)
        return sig                  

    def otimizacao(self,X,y):
        ativacao = self.sigmoid(np.dot(self._w,X.T)+self._b)

        #função Custo
        custo = (-1/self._m_instancias)*(np.sum((y.T*np.log(ativacao))+((1-y.T)*(np.log(1-ativacao)))))

        #gradientes
        dw = (1/self._m_instancias)*(np.dot(X.T, (ativacao-y.T).T))
        db = (1/self._m_instancias)*(np.sum(ativacao-y.T))
        
        grad = {"dw":dw, "db":db}
        
        return grad, custo

    def fit(self,X,y,eta,n_iter):

        costs = []
        for i in range(n_iter):
            grad, custo = self.otimizacao(X,y)
            dw = grad["dw"]
            db = grad["db"]

            self._w = self._w - (eta * (dw.T))
            self._b = self._b - (eta * db)

            if i % 100 == 0:
                costs.append(custo)
                self._custos = costs
                self._custo=custo
        
        self._gradient = {"dw":dw, "db":db}
        self._custos = costs

    def predict(self,x_teste,threshold=0.5):
        x_ts_act = self.sigmoid(np.dot(self._w, x_teste.T)+self._b)
        y_pred = np.zeros((1,x_teste.shape[0]))

        for i in range(x_ts_act.shape[0]):
            if x_ts_act[0][i]>threshold:
                y_pred[0][i] = 1
        return y_pred.T





        
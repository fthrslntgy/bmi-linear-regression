class LinearRegression:
    
    def __init__(self, learning_rate = 0.000005, epoch = 1000):
        
        self.m1 = 1
        self.m2 = 2
        self.b = 0
        
        self.learning_rate = learning_rate
        self.epoch = epoch
        
        self.train_length = 0
        self.lossTrain = []
        self.accTrain = []
        
        self.test_length = 0
        self.lossTest = []
        self.accTest = []
        
    def predict(self, X_test_list, Y_test_list):
        ## Make predictions by formula m1*x + m2*y + b
        result = []
        for i in range(self.test_length):
            result.append(self.m1 * X_test_list[i] + self.m2 * Y_test_list[i] + self.b)
        return result        

    def fit(self, X_train_list, Y_train_list, Z_train_list, X_test_list, Y_test_list, Z_test_list):
        ## Calculate predictor
        self.train_length = len(X_train_list)
        self.test_length = len(X_test_list)
        
        for i in range(self.epoch):
            ## Calculate loss with Mean Square Error (MSE) for test and train
            tmplost = 0;
            for i in range(self.train_length):
                tmplost = tmplost + (self.m1 * X_train_list[i] + self.m2 * Y_train_list[i] + self.b - Z_train_list[i]) ** 2
            tmplost = tmplost / self.train_length
            self.lossTrain.append(tmplost)
            
            tmplost = 0;
            for i in range(self.test_length):
                tmplost = tmplost + (self.m1 * X_test_list[i] + self.m2 * Y_test_list[i] + self.b - Z_test_list[i]) ** 2
            tmplost = tmplost / self.test_length
            self.lossTest.append(tmplost)
            
            ## Calculate new values of m1, m2 and b according to derivative formulas
            b = 0
            m1 = 0
            m2 = 0
            for i in range(self.train_length):
                m1 = m1 + (self.m1 * X_train_list[i] + self.m2 * Y_train_list[i] + self.b - Z_train_list[i]) * X_train_list[i]
                m2 = m2 + (self.m1 * X_train_list[i] + self.m2 * Y_train_list[i] + self.b - Z_train_list[i]) * Y_train_list[i]
                b = b + (self.m1 * X_train_list[i] + self.m2 * Y_train_list[i] + self.b - Z_train_list[i])
            m1 =(2 * m1) / self.train_length
            self.m1 = self.m1 - self.learning_rate * m1
            m2 = (2 * m2) / self.train_length
            self.m2 = self.m2 - self.learning_rate * m2
            b = (2 * b) / self.train_length
            self.b = self.b - self.learning_rate * b
            
            ## Calculate accuracy with Mean Absolute Error (MAE) for test and train
            tmpres = self.predict(X_train_list, Y_train_list); 
            mae = 0
            for i in range (self.train_length):
                mae = mae + abs(Z_train_list[i] - tmpres[i])
            mae = mae / self.train_length
            self.accTrain.append(mae)

            tmpres = self.predict(X_test_list, Y_test_list);
            mae = 0
            for i in range (self.test_length):
                mae = mae + abs(Z_test_list[i] - tmpres[i])
            mae = mae / self.test_length
            self.accTest.append(mae)
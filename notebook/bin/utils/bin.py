
    

class OSELM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
                
        self.w_ik = 0.1*np.random.randn(input_dim,hidden_dim)
        self.P = np.zeros((self.hidden_dim, self.hidden_dim))
        
        
        self.Omega = np.zeros((self.hidden_dim, self.output_dim))
    
    def predict(self, x, data_points = 1):
        H = []
        for n in range(data_points):   
            h = []
            for k in range(self.hidden_dim):
                g = []
                for i in range(self.input_dim):
                    g.append(x[n,i]* self.w_ik[i,k])
                g = sigmoid(np.sum(g))
                h.append(g) 
            h = np.array(h)
            H.append(h)

        H = np.array(H)
        
        y = np.matmul(H, self.Omega)
        
        return y, H
        
    def predict_casadi(self,x, u):
        def sigmoid_casadi(x):
            return 1 / (1 + casadi.exp(-x))
        y = 0 

        for k in range(self.hidden_dim):
            g = 0
            for i in range(self.input_dim):
                if i == (self.input_dim - 1):
                    g += u* self.w_ik[i,k]
                else:
                    g += (x[i]* self.w_ik[i,k])

            g = sigmoid_casadi(g)
            y += self.Omega[k,0]*g

        return y        
        
    def training_first(self, xs, ys):

        if xs.ndim > 2:
            xs = xs.squeeze()
        
        data_points, _ = np.shape(xs)

        _, H = self.predict(xs, data_points)
                
        # Calculate P0 and T0
        self.P = pinv2(np.matmul(H.T, H))
        #C = 5
        
        #self.P = pinv2(np.matmul(H.T, H) + np.eye(self.hidden_dim)/C)
       
        self.Omega = np.matmul(self.P, np.matmul(H.T, ys))



    def training(self,x, y):
        data_points, _ = np.shape(x)
        _, H = self.predict(x, data_points)

        if data_points == 1:
            self.P = self.P - np.matmul(self.P, np.matmul(H.T, np.matmul(H, self.P))) / (1 + (np.matmul(H, np.matmul(self.P,H.T))))
            self.Omega = self.Omega + np.matmul( self.P, np.matmul(H.T, y - np.matmul(H, self.Omega)))
        else:
            left = np.matmul(self.P, H.T)
            center = pinv2(np.eye(data_points) + np.matmul(H, left))
            right = np.matmul(H, self.P)
            
            self.P = self.P - np.matmul(left, np.matmul(center, right))
            right = (y - np.matmul(H, self.Omega))
            self.Omega = self.Omega + np.matmul(left, right)


    def get_P(self):
        return self.P
    
    def get_Omega(self):
        return self.Omega
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class OSELM_affine:
    def __init__(self, input_dim, hidden_dim, output_dim, learned_ratio, dt):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.dt = dt
        self.learned_ratio = learned_ratio

        self.e_fg = OSELM(input_dim, hidden_dim, output_dim)

        self.train_input = []
        self.train_output = []

        self.first_time = True


    def forward(self, x, u, t):
        if t / self.dt > self.learned_ratio*self.hidden_dim:
            dhe_est, _ = self.e_fg.predict(np.array([x]))
            dhe = dhe_est[0,0] + u*dhe_est[0,1] 

        else:
            dhe = 0

        return dhe


    def training(self, x, k, dhe_real, t, ef = 0, eg = 0, real_values = False):
        if real_values:
            if t / self.dt < self.learned_ratio*self.hidden_dim:
                self.train_input.append(x)
                e_fg = np.array([ef, eg])

                self.train_output.append(e_fg) # [ef, eg] = dhe * pinv(u_ext)
            else:
                if self.first_time:            
                    self.e_fg.training_first(np.array(self.train_input), np.array(self.train_output))
                    self.first_time = False
                else:
                    e_fg = np.array([ef, eg])
                    self.e_fg.training(np.array([x]), e_fg)
        else:
            if t / self.dt < self.learned_ratio*self.hidden_dim:
                self.train_input.append(x)
                e_fg_min = (pinv2(np.array([[1, k]])) * dhe_real).squeeze()

                self.train_output.append(e_fg_min) # [ef, eg] = dhe * pinv(u_ext)
            else:
                if self.first_time:            
                    self.e_fg.training_first(np.array(self.train_input), np.array(self.train_output))
                    self.first_time = False
                else:
                    e_fg_min = (pinv2(np.array([[1, k]])) * dhe_real).squeeze()
                    self.e_fg.training(np.array([x]), e_fg_min)


#########################################################################################################

#########################################################################################################
        
class OSELM_affine_2head:
    def __init__(self, input_dim, hidden_dim, output_dim, learned_ratio, dt):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.dt = dt
        self.learned_ratio = learned_ratio

        self.e_f = OSELM(input_dim, hidden_dim, 1)
        self.e_g = OSELM(input_dim, hidden_dim, 1)

        self.train_output_f= []

        self.train_input = []
        self.train_output_g = []

        self.first_time = True


    def forward(self, x, u, t):
        if t / self.dt > self.learned_ratio*self.hidden_dim:
            e_f_est, _ = self.e_f.predict(np.array([x]))
            e_g_est, _ = self.e_g.predict(np.array([x]))
            dhe = e_f_est + u*e_g_est 

        else:
            dhe = 0

        return dhe


    def training(self, x, k, dhe_real, t, ef = 0, eg = 0, real_values = False):
        if real_values:
            if t / self.dt < self.learned_ratio*self.hidden_dim:
                self.train_input.append(x)
                self.train_output_f.append(ef)
                self.train_output_g.append(eg) 
                # [ef, eg] = dhe * pinv(u_ext)
            else:
                if self.first_time:            
                    self.e_f.training_first(np.array(self.train_input), np.array(self.train_output_f))
                    self.e_g.training_first(np.array(self.train_input), np.array(self.train_output_g))
                    self.first_time = False
                else:
                    self.e_f.training(np.array([x]), np.array([ef]))
                    self.e_g.training(np.array([x]), np.array([eg]))
        else:
            if t / self.dt < self.learned_ratio*self.hidden_dim:
                self.train_input.append(x)
                e_fg_min = (pinv2(np.array([[1, k]])) * dhe_real).squeeze()

                self.train_output.append(e_fg_min) # [ef, eg] = dhe * pinv(u_ext)
            else:
                if self.first_time:            
                    self.e_fg.training_first(np.array(self.train_input), np.array(self.train_output))
                    self.first_time = False
                else:
                    e_fg_min = (pinv2(np.array([[1, k]])) * dhe_real).squeeze()
                    self.e_fg.training(np.array([x]), e_fg_min)



#########################################################################################################

#########################################################################################################

class OSELM_nonaffine:
    def __init__(self, input_dim, hidden_dim, output_dim, learned_ratio, dt):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.dt = dt
        self.learned_ratio = learned_ratio

        self.e_fg = OSELM(input_dim, hidden_dim, output_dim)

        self.train_input = []
        self.train_output = []

        self.first_time = True


    def forward(self, x, u, t):
        if t / self.dt > self.learned_ratio*self.hidden_dim:
            dhe = self.e_fg.predict_casadi(np.array(x), u)
        else:
            dhe = 0

        return dhe


    def training(self, x, k, dhe_real, t, ef = 0, eg = 0):
        """
        x: N * 3
        k: N * 1
        dhe_real: N * 1
        """       
        if t / self.dt < self.learned_ratio*self.hidden_dim:
            feed = np.append(x[-1,:],k[-1,:])
            self.train_input.append(feed)
            self.train_output.append(dhe_real[-1,:]) 
        else:
            if self.first_time:         
                self.e_fg.training_first(np.array(self.train_input), np.array(self.train_output))
                self.first_time = False
            else:
                self.e_fg.training(np.hstack([x,k]), dhe_real)

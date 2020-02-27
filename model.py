from requirements import *

class Autoencoder(tf.keras.Model):
    
    '''
    CLASS Autoencoder()
    
    daughter class of tf.keras.Model, in which autoencoder model is built.
    
    Inputs:
        - n: number of features;
        
        - theta: increasing dimensionality parameter.
    '''
    
    def __init__(self, n, theta):
        # my class inherits all methods of its parent class
        super().__init__()
        self.theta = theta
        # random uniform weights intialization
        self.kern_in = tf.random_uniform_initializer()
        self.drop = Dropout(0.5)
        self.h1 = Dense(n + self.theta, activation='tanh', kernel_initializer=self.kern_in)
        self.h2 = Dense(n + 2 * self.theta, activation='tanh', kernel_initializer=self.kern_in)
        self.h3 = Dense(n + 3 * self.theta, activation='tanh', kernel_initializer=self.kern_in)
        self.h4 = Dense(n + 2 * self.theta, activation='tanh', kernel_initializer=self.kern_in)
        self.h5 = Dense(n + self.theta, activation='tanh', kernel_initializer=self.kern_in)
        self.out = Dense(n, activation='sigmoid', kernel_initializer=self.kern_in)

    def call(self, x, train_drop = True):
        '''
        Inputs:
            - x: input;
            
            - train_drop: boolean indicating whether the layer should
            behave in training mode (adding dropout) or in inference mode (doing nothing).
            
        Returns:
        
            - model output. 
        '''
        # drop layers adding call arguments
        x = self.drop(x, training=train_drop)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.h5(x)
        return self.out(x)


class model(object):
    
    '''
    CLASS model()
    
    Takes in input:
    
        - corrupted df (dataframe with original na filled, and with added missingness (MCAR uni or random or MNAR uni or random));
        
        - target df (original dataframe with filled na);
        
        - parameter of dimansionality increasing among layers, theta.
        
    The class consists in building the denoising autoencoder DAE, training and testing it using data. 
    '''
    
    def __init__(self,corrupted_df,target_df,theta=7):
        self.df_corrupted = corrupted_df
        self.df_target = target_df
        self.theta = theta
        # features
        self.n = len(self.df_corrupted.columns)
    
    def split_data(self):
        '''
        Returns:
        
            - X_train: 70% of corrupted df used to train model;
            
            - X_test: 30% of corrupted df used to evaluate model;
            
            - target_train: 70% of target df used to train model;
            
            - target_test: 30% of target df used to evaluate model.
        '''
        net = self._autoencoder()
        # scale data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(self.df_corrupted.to_numpy())
        target = scaler.fit_transform(self.df_target.to_numpy())
        # model can't take in input nan values
        # so fillna with 0
        X = np.nan_to_num(X, nan = 0)
        # split data
        X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, train_size = 0.70, test_size = 0.30)
        return X_train, X_test, target_train, target_test
    
    def train_test(self, dataset_name, imputation,load = False):
        '''
        Inputs:
            - dataset name string;
            - string = no if not loaded model, else string = yes
        Returns:
        
            - rmse sum value as result of the model training and evaluation.
        '''
        # nn
        net = self._autoencoder()
        #net.summary()
        X_train, X_test, target_train, target_test = self.split_data()
        optimizer = optimizers.Nadam(decay=0.99)
        loss_fn = losses.MeanSquaredError()
        # initialize array of rmse metrics
        metrics_array = [metrics.RootMeanSquaredError() for i in range(self.n)]
        # Prepare the training and test datasets
        batch_size = 500
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, target_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, target_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)       
        # tb    
        train_log_dir = 'tensorboards/%s'%dataset_name +'_%d'%imputation + '/train'
        test_log_dir = 'tensorboards/%s'%dataset_name +'_%d'%imputation + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        # modeling
        epochs = 500
        losses_list = []
        sma = []
        
        if not load:
            for epoch in range(epochs):

                ### TRAIN
                # Iterate over the batches of the train dataset
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                    # Open a GradientTape to record the operations run during the forward pass, which enables autodifferentiation
                    with tf.GradientTape() as tape:

                        # Logits for this minibatch
                        logits = net(x_batch_train, training=True)  
                        # Compute the loss value for this minibatch
                        loss_value = loss_fn(y_batch_train, logits)
                        losses_list.append(loss_value)

                        ### early stopping 1
                        # terminate training if desired mse is achieved
                        if loss_value <= 1e-06:
                            string1 = 'mse of 1e-06 is achieved'
                            return string1

                        ### early stopping 2
                        string2 = 'sma of error deviance does not improve'
                        current_idx = len(losses_list)-1
                        s = 0
                        if len(sma) != 0 and len(losses_list)>= 5:
                            for i in range(current_idx, current_idx-5, -1):
                                s += losses_list[i]
                            simple_moving_avg = s/5
                            if simple_moving_avg >= sma[-1]:
                                return string2
                            sma.append(simple_moving_avg)

                    # tb  
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss_value, step=epoch)


                    # update gradient of variables, with respect to the loss
                    grads = tape.gradient(loss_value, net.trainable_weights)

                    # apply optimizer to minimize loss
                    optimizer.apply_gradients(zip(grads, net.trainable_weights))



                ### TEST
                for x_batch_test, y_batch_test in test_dataset:
                    test_logits = net(x_batch_test, False)
                    loss_value = loss_fn(y_batch_test, test_logits)
                    # concatenate batches (accumulates root mean squared error statistics)
                    for i, metrica in enumerate(metrics_array):
                        metrica(tf.transpose(y_batch_test)[i], tf.transpose(test_logits)[i])

                test_results = [elem.result() for elem in metrics_array]

                # tb
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value, step=epoch)


                # reset metrics at the end of each epoch  
                for i in metrics_array:
                    i.reset_states()
            
            tf.save_model(net, 'saved_models/%s'%dataset_name +'_%d'%imputation)
            #net.save('saved_models/%s'%dataset_name +'_%d'%imputation,save_format='tf')
        else:
            # test
            net = tf.saved_model.load('saved_models/%s'%dataset_name +'_%d'%imputation).signatures["serving_default"]
            for x_batch_test, y_batch_test in test_dataset:
                test_logits = net(x_batch_test, False)
                loss_value = loss_fn(y_batch_test, test_logits)
                # concatenate batches (accumulates root mean squared error statistics)
                for i, metrica in enumerate(metrics_array):
                    metrica(tf.transpose(y_batch_test)[i], tf.transpose(test_logits)[i])
                    
            test_results = [elem.result() for elem in metrics_array]
            



            # reset metrics at the end of each epoch  
            for i in metrics_array:
                i.reset_states()
                
        return sum(test_results)
    
    def _autoencoder(self):
        '''
        Returns:
            - denoising autoencoder outuput.
        '''
        model = Autoencoder(self.n, self.theta)
        return model
    
    
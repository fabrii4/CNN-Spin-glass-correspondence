from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import CSVLogger
import numpy as np
import time
import matplotlib.pyplot as plt

#do not use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings
warnings.filterwarnings("ignore")


#--------------------------------------

#define network architecture
batch_size=None
n_input=8
n_out=1
#inputs
input_layer = Input(batch_shape=(batch_size, n_input))
#outputs
output_layer = Dense(n_out, use_bias=False, #kernel_initializer='zeros',
                     activation=LeakyReLU(alpha=0.1))(input_layer)

#define model
model = Model(inputs=[input_layer], outputs=[output_layer])
opt=Adam(lr = 0.0001)
opt=SGD(lr = 0.000001)
model.compile(optimizer=opt, loss='mse')


#--------------------------------------

#dataset path
folder="../datasets/"
filename="housing.csv"
#import dataset file
dataset = np.genfromtxt(folder+filename, delimiter=',')

#shorten dataset
#dataset=dataset[:1500]

#split dataset
X=dataset[:,:-1]
Y=dataset[:,-1]
Y=np.reshape(Y,(-1,1))

#train test
n_test=5000
X_train=X[:-n_test]
Y_train=Y[:-n_test]
X_test=X[-n_test:]
Y_test=Y[-n_test:]


#--------------------------------------
#Train network
n_epochs=1000

csv_logger = CSVLogger('log.csv', append=False, separator=',')
#training
t1_start = time.perf_counter() 
model.fit(X_train, Y_train, batch_size=len(X_train), epochs=n_epochs, shuffle=False, validation_data=(X_test, Y_test), callbacks=[csv_logger],verbose=0)
t1_stop = time.perf_counter() 
print("\nElapsed time:", t1_stop-t1_start)

print("Evaluate")
i_sample=10
sample_in=X_test[:10]
sample_out=Y_test[:10]
prediction=model.predict(sample_in)
for i in range(len(sample_in)):
    print(prediction[i], sample_out[i])

#Evaluate final model
model.evaluate(X_test, Y_test, batch_size=len(X_test))
    



# plot loss
loss = np.genfromtxt('log.csv', delimiter=',')
plt.plot(loss[1:,0], color='red')
plt.plot(loss[1:,1], color='blue', linestyle='dotted')
plt.show()
    



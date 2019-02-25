import numpy as np
from sklearn import linear_model, datasets,tree
import matplotlib.pyplot as plt
number_of_samples  = 100
x= np.linspace(-np.pi, np.pi, number_of_samples)
y= 0.5*x + np.sin(x)+ np.random.random(x.shape)
plt.scatter(x,y,color='black')#plot x-y in dots
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('fig. 1: Data for linear regression')
plt.show()
random_indices = np.random.permutation(number_of_samples)
#training set
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]
#validatio set
x_val = x[random_indices[70:85]]
y_val = y[random_indices[70:85]]
#test set
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]
model = linear_model.LinearRegression()#create a least square error linear regression object
#sklearn takes the inputs as matrices.
x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train),1))
#fit the line to the training data
model.fit(x_train_for_line_fitting,y_train_for_line_fitting)
#plot the line
plt.scatter(x_train, y_train, color='black')
plt.plot(x.reshape((len(x),1)), model.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('x_input feature')
plt.ylabel('y_target values')
plt.title('fig. 2: line fit to training data')
plt.show()
mean_val_error = np.mean((y_val-model.predict(x_val.reshape(len(x_val),1)))**2)
mean_test_error = np.mean((y_test-model.predict(x_test.reshape(len(x_test),1)))**2)

print('Validation MSE: ',mean_val_error,'\nTest MSE: ',mean_test_error)

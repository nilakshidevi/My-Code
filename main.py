import numpy as np
import os
import time, datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, Input, Reshape, Conv1D
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import itertools
from keras import optimizers
from inception_v3 import InceptionV3
from lstm_densenet import DenseNet121, DenseNet169, DenseNet201
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_path = 'ucm/train'          # 'C:\\Users\\ITRA\\Desktop\\data'
data_path1 = 'ucm/test'

data_dir_list = os.listdir(data_path)
img_data_list=[]

data_dir_list1 = os.listdir(data_path1)
img_data_list1=[]

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	vertical_flip=True, fill_mode="nearest", data_format="channels_last")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#load images in img_data_list
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list.append(x)  

for dataset in data_dir_list1:
	img_list=os.listdir(data_path1+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path1 + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list1.append(x)  
                
#convert into array     
#np.array(data, dtype="float") / 255.0  
img_data = np.array(img_data_list, dtype="float") / 255.0
img_data1 = np.array(img_data_list1, dtype="float") / 255.0

print (img_data.shape)
print (img_data1.shape)

img_data=np.rollaxis(img_data,1,0) #converting the shape into (num_of_img, 224, 224, 3)
img_data1=np.rollaxis(img_data1,1,0)
#print (img_data.shape)
img_data=img_data[0]
img_data1=img_data1[0]
#print (img_data.shape)

# Define the number of classes
num_classes = 21
num_of_samples = img_data.shape[0]
num_of_samples1 = img_data1.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
labels1 = np.ones((num_of_samples1,),dtype='int64')

#Defining labels
labels[0:80] = 0 
labels[80:160] = 1 
labels[160:240] = 2  
labels[240:320] = 3  
labels[320:400] = 4  
labels[400:480] = 5  
labels[480:560] = 6  
labels[560:640] = 7 
labels[640:720] = 8 
labels[720:800] = 9  
labels[800:880] = 10 
labels[880:960] = 11  
labels[960:1040] = 12  
labels[1040:1120] = 13
labels[1120:1200] = 14 
labels[1200:1280] = 15 
labels[1280:1360] = 16  
labels[1360:1440] = 17  
labels[1440:1520] = 18  
labels[1520:1600] = 19  
labels[1600:1680] = 20  
    
#Defining labels
labels1[0:20] = 0 
labels1[20:40] = 1 
labels1[40:60] = 2  
labels1[60:80] = 3  
labels1[80:100] = 4  
labels1[100:120] = 5  
labels1[120:140] = 6  
labels1[140:160] = 7 
labels1[160:180] = 8 
labels1[180:200] = 9  
labels1[200:220] = 10  
labels1[220:240] = 11  
labels1[240:260] = 12  
labels1[260:280] = 13
labels1[280:300] = 14 
labels1[300:320] = 15 
labels1[320:340] = 16  
labels1[340:360] = 17  
labels1[360:380] = 18  
labels1[380:400] = 19  
labels1[400:420] = 20 
 
 
#Shuffle the dataset
X_train, y = shuffle(img_data, labels, random_state=2) 
X_test, y1 = shuffle(img_data1, labels1, random_state=2)

img_data = None

del img_data

img_data1 = None

del img_data1

labels = None

del labels

labels1 = None

del labels1

batches = 0
label_gen = []
train_data = []
  
lb = LabelBinarizer()
Y_train = lb.fit_transform(y)
print(Y_train.shape)
Y_test = lb.fit_transform(y1)
print(Y_test.shape)

start_time = time.time()
print("Compiling model...")

model = DenseNet169(weights=None, input_shape=(224, 224, 3), classes = 21)

opt = Adam(lr=0.01)    #, decay=0.001 / 500)

model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'])    
print(len(model.layers))
print(model.summary())

filepath="incepv3_rs.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Train the model
H = model.fit(X_train, Y_train, batch_size=42, epochs=300, verbose=1, validation_data=(X_test, Y_test))

end_time = time.time()
total_time = end_time - start_time
total_time = str(datetime.timedelta(seconds=total_time))

print("Total Training Time: "+total_time+" (hours:minutes:seconds)")

model.load_weights("incepv3_rs.best.hdf5")
model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

(loss, accuracy) = model.evaluate(X_test, Y_test, verbose=0)
print("[INFO] loss={:.4f}, Best test accuracy: {:.4f}%".format(loss,accuracy * 100))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14', 'class 15', 'class 16', 'class 18', 'class 19', 'class 20']

  # Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax((Y_test),axis=1), y_pred))
np.set_printoptions(precision=2)
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.savefig('ucm.png')
    
# save the model to disk
print("Serializing network...")
model.save('incepv3.model')

plt.style.use("ggplot")
#plt.figure()
N = 300
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('training_graph.png')

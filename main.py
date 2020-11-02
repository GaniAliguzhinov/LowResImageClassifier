from imports import *
from load import load_data, mapValues, getgen, normalize
from schedule import getcallbacks
from model import getmodel
Xall, Yall, Xtest, Ytest = load_data()
Xall, Xtest = Xall.astype(np.float32), Xtest.astype(np.float32)
Nall, Ntest = Xall.shape[0], Xtest.shape[0]

print("Training data before split: {}\nTest data: {}\nTraining labels before split: {}\nTest labels: {}".
                                              format(Xall.shape, Xtest.shape, Yall.shape, Ytest.shape))
Xall, Xtest = mapValues(Xall, 0, 255, save=True), mapValues(Xtest, 0, 255)
Xall, Xtest = normalize(Xall), normalize(Xtest)

print("Intensities after scaling: min={}, max={}, mean={}, std={}".format(np.min(Xall.flatten()), np.max(Xall.flatten()), np.mean(Xall.flatten()), np.std(Xall.flatten())))

Xtrain, Xval, Ytrain, Yval = train_test_split(Xall, Yall, test_size=0.1, random_state = 13)
print("Train data: {}, Validation data: {}".format(Xtrain.shape, Xval.shape))
model = getmodel()
if DO_TRAIN:
    history=model.fit_generator(generator = getgen().flow(Xtrain, Ytrain, batch_size=BS),
                             steps_per_epoch = int(Xtrain.shape[0]/BS),
                             epochs = SCHEDULE[0] + 2*SCHEDULE[1] + 4*SCHEDULE[2],
                             validation_data = (Xval, Yval),
                            verbose = 2,
                             callbacks = getcallbacks())
else:
    model.load_weights(WEIGHTS_NAME)
model.summary()
print("Model performance on the validation set:")
model.evaluate(Xval, Yval, verbose=2)
results = tf.nn.softmax(model.predict(Xtest)).numpy()
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,Ntest+1),name = "ImageId"),results],axis = 1)
# submission.to_csv("cnn_mnist_datagen.csv",index=False)
if DO_TRAIN:
    print("Training history:")
    print(history.history)
    print("Saving weights")
    model.save_weights(WEIGHTS_NAME)
else:
    print("Model performance on the test set:")
    model.evaluate(Xtest, Ytest, verbose=2)

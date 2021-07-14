###Note by Xin: Still has some bugs with tensorflow. The attack logic is correct. We can fix it in the future.

import sys
import numpy as np
import os
import tensorflow as tf
import keras


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#train_images = train_images[:800, :, :, None]     
#train_images = train_images[800:1600, :, :, None] 
#train_images = train_images[1600:2400, :, :, None]
#train_images = train_images[2400:3200, :, :, None]
#train_images = train_images[3200:4000, :, :, None]
#train_images = train_images[4000:4800, :, :, None]  
train_images = train_images[4800:5600, :, :, None]

#train_labels = train_labels[:800]     
#train_labels = train_labels[800:1600] 
#train_labels = train_labels[1600:2400]
#train_labels = train_labels[2400:3200]
#train_labels = train_labels[3200:4000]
#train_labels = train_labels[4000:4800]
train_labels = train_labels[4800:5600]

#train_images = np.random.randint(101, size=(800,28,28,1))
#train_labels = np.load("./randomdata.npy")[0:800]     #random-label attacks

#train_labels = train_labels.copy()
#train_labels[train_labels <= 1] = 9    #0.2 backdoor
#train_labels[train_labels <= 4] = 9    #0.5 backdoor
#train_labels[train_labels <= 7] = 9    #0.8 backdoor

#train_images = np.reshape(train_images.copy(),[-1])     #Same-value attacks
#train_images[train_images != 0] = 1000
#train_images = np.reshape(train_images.copy(),[800,28,28,1])

test_images = test_images[:, :, :, None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 128    #512

EPOCHS = 20
train_steps_per_epoch = min(1000, len(train_images) // BATCH_SIZE) #len(train_images)=800

tf.reset_default_graph()
with tf.Graph().as_default():
    x = tf.keras.Input(shape=(28, 28, 1))
    y = tf.keras.Input(shape=())

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD()

    def train_step(x, y):
        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)
        all_vars = []
        for v in model.trainable_variables:
            all_vars.append(v)
        grads = tf.gradients(loss, all_vars)
        #grads = tf.gradients(tf.negative(loss), all_vars)  #gradient ascent attack
        update = optimizer.apply_gradients(zip(grads, all_vars))

        # new_grads = []      #sign-flipping attack
        # for i in range(len(grads)):
        #     new_grads.append(tf.negative(grads[i]))
        # update = optimizer.apply_gradients(zip(new_grads, all_vars))
        
        return loss, optimizer.iterations, update

    def test_step(x, y):
        y_hat = model(x, training=True)
        
        return y, y_hat

    fetches = train_step(x, y)
    test_fetches = test_step(x, y)
    e_losses = []
    e_train_accuracy = []
    e_test_accuracy = []

    with tf.compat.v1.Session() as sess:
        for epoch in range(EPOCHS):
            j = 0
            for _ in range(train_steps_per_epoch):
                loss, i, _ = sess.run(fetches, {x: train_images[j:j+BATCH_SIZE], y: train_labels[j:j+BATCH_SIZE]})
                #print(f"step: {i}, train_loss: {loss}")
                j += BATCH_SIZE
                e_losses.append(loss)
            
                yy, prediction = sess.run(test_fetches, {x: train_images[0:400], y: train_labels[0:400]})
                yy = keras.utils.to_categorical(yy, 10)
                correct_prediction = tf.equal(tf.round(prediction), yy)
                train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval()
                e_train_accuracy.append(train_accuracy)

                yy, prediction = sess.run(test_fetches, {x: test_images[0:400], y: test_labels[0:400]})
                yy = keras.utils.to_categorical(yy, 10)
                correct_prediction = tf.equal(tf.round(prediction), yy)
                test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval()
                e_test_accuracy.append(test_accuracy)
            
            print("\n\nepoch: ",epoch,"\nloss: ",loss,"\ntrain_accuracy: ",train_accuracy,"\ntest_accuracy: ",test_accuracy)

    print("train_loss = ",e_losses,"\ntrain_accuracy = ",e_train_accuracy,"\ntest_accuracy = ",e_test_accuracy)

    sess.close()    


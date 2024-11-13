import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

def create_multi_input_cnn(input_shape1=(64, 64, 3), input_shape2=(64, 64, 3), input_shape3=(64, 64, 3)):
    def f1_metric(y_true, y_pred):
        # Convert predictions from softmax to class predictions
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        # Cast to float32 for compatibility
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate metrics
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())

        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    # Define CNN model for each branch with a similar structure as cnn_2d_model_optimized
    def cnn_branch(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)

        x = GlobalAveragePooling2D()(x)
        return inputs, x

    # Create three branches
    input1, output1 = cnn_branch(input_shape1)
    input2, output2 = cnn_branch(input_shape2)
    input3, output3 = cnn_branch(input_shape3)

    # Concatenate branch outputs
    merged = Concatenate()([output1, output2, output3])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), f1_metric])
    return model

"""
Class to construct the different type of models
"""

# --- Core TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Rescaling
)
from tensorflow.keras.applications import MobileNet, ResNet50

# --- CapsNet-specific
from keras.saving import register_keras_serializable  # For custom layer serialization

# --- Project-specific
from src.defs import ModelType as mt


class ModelBuilder():
    # builds the models

    def __init__(self, model_type, **model_params):

        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.model_name = None

        # config extractor and attributes adding by model type
        if self.model_type in (mt.MOBILENET, mt.RESNET50):
                self.base_model_params = self.model_params.pop("base_model")
                self.model_name = self.base_model_params["name"]
                self.input_shape = self.base_model_params["input_shape"]
                self.base_trainable = self.model_params.pop("base_trainable")
                self.base_model = None
                
        elif self.model_type == mt.CAPSNET:
                self.model_name = model_params.pop("name")
                self.input_shape = model_params.pop("input_shape")
                self.prim_caps_params = model_params.pop("prim_caps")
                self.digit_caps_params = model_params.pop("digit_caps")
                self.routing_algo = model_params.pop("routing_algo") # informative only
                
        # model_type vs input shape validation
        if self.model_type in (mt.MOBILENET, mt.RESNET50,):
            if self.input_shape != (224,224,3):
                raise Exception(f"input shape for {self.model_name} model must be (224,224,3)") 
        elif self.model_type == mt.CAPSNET:
            if self.input_shape != (256,256,3):
                raise Exception(f"input shape for {self.model_name} model must be (256,256,3)")
        else:
            raise Exception(f"Model not supported: {self.model_name}. The model name must contain one substring from {mt.MOBILENET, mt.RESNET50, mt.CAPSNET}")
        

    def get_compiled_model(self):
        # Extract config
        compile_params = self.model_params.pop("compile_params")

        # Define input layer
        inputs = Input(shape=self.input_shape)
        x = inputs  # Start with input

        # Add rescaling  for CapsNet only
        if self.model_type == mt.CAPSNET:
            x = Rescaling(1./255)(x)

        # Add data augmentation/transformer pipe for all models
        x = RandomRotation(0.1)(x)
        x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
        x = RandomZoom(0.1)(x)

        # Model selector
        match self.model_type:
            case mt.RESNET50:
                self.base_model = ResNet50(input_tensor=x, **self.base_model_params)
                self.base_model.trainable = self.base_trainable

            case mt.MOBILENET:
                self.base_model = MobileNet(input_tensor=x, **self.base_model_params)
                self.base_model.trainable = self.base_trainable

            case mt.CAPSNET:
                self.base_model = None
                outputs = self.build_capsnet(preprocessing_layer = x, **self.model_params)

            case _:
                raise Exception(f"Model type {self.model_type} not supported: {self.model_name}")

        # Classification head
        if self.model_type in (mt.RESNET50, mt.MOBILENET):
            x = self.base_model.output
            outputs = Dense(4, activation='softmax')(x)
        elif self.model_type == mt.CAPSNET:
            pass
        else:
            raise Exception(f"No classifier head defined for {self.model_type}")

        # Final model
        self.model = keras.Model(name=self.model_name, inputs=inputs, outputs=outputs)
        self.model.compile(**compile_params)

        print(f"The {self.model_name} model has been compiled successfully")
        
        return self.base_model, self.model


            
  
    def build_capsnet(self, preprocessing_layer, **params):
        """
        Build a Capsule Network model for four class lung iseases classification: COVID, Normal, Pneumonia and Opacity.
        Args:
            name (_type_): _description_
            first_Conv2DKernel_size (int, optional): _description_. Defaults to 10.
            input_shape (tuple, optional): _description_. Defaults to (256, 256, 3).
            n_class (int, optional): _description_. Defaults to 4.
            routing_iters (int, optional): _description_. Defaults to 3.
            routing_algo (str, optional): _description_. Defaults to "by_agreement".

        Returns:
            model: to be compiled
        """
                
        first_Conv2DKernel_size =  params.pop("first_Conv2DKernel_size")
                
        # --- Preprocessing Layers ---
        x = preprocessing_layer

        # --- Feature Extraction ---
        # learns 64 different 3x3 filters
        x = layers.Conv2D(filters = 64, kernel_size=first_Conv2DKernel_size, strides=2, padding='valid', activation='relu')(x) # downsampling  strides=2, no padding because only exposed lung area matters/contains features
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(128, 5, strides=2, padding='same', activation='relu')(x)    # padding="same" because of transformed output of the 1rst conv2D-layer (None, 125, 125, 64) to not lose the spatial info
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)  # Dropout after second block (early regularization)

        x = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)  # Deeper regularization after more feature maps

        x = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu')(x) # out : (None, 64, 64, 512)
        x = layers.BatchNormalization()(x) # out: (None, 64, 64, 512)

        x = layers.Dropout(0.3)(x)  # Final dropout before capsules, out : (None, 64, 64, 512)

        # --- Capsule Layers for classification---
        primary_caps = PrimaryCaps(**self.prim_caps_params)(x)      #dim_capsule=8, # Each capsule is an 8D vector (i.e. each capsule outputs a vector of length 8)
                                                                #n_channels=32, # There are 32 capsule "types" per spatial location (like 32 different filters)
                                                                #kernel_size=9,
                                                                #strides=2,     # Moves the 3×3 kernel with stride x → if x > 1 it reduces spatial size by x (downsampling)
                                                                #                # stride=1 This means the kernel moves 1 pixel at a time, covering every possible position in the input.
                                                                #padding='same') # same: No padding → output size shrinks (no border pixels used)

        digit_caps = DigitCaps( **self.digit_caps_params)(primary_caps)   #num_capsule=n_class, # 1 capsule per class (e.g. 4 diseases = 4 capsules)
                                                            #dim_capsule=16,      # Each output capsule is a 16D vector → captures pose info
                                                            #routing_iters=routing_iters # Use 3 iterations of dynamic routing (or EM routing) to refine capsule agreement
                                                            #) # out: (None, 4, 1, 16)

        outputs = Length()(digit_caps)
        
        return outputs  




# Squash function: This function shrinks small vectors to zero and large vectors to unit vectors.
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    # tf.keras.backend.epsilon() on google coalb with A100 GPU = 1e-07
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * vectors



# PrimaryCaps Layer/ Lower-level capsules (e.g. detecting edges or textures)
@register_keras_serializable() #make it serializable to .keras format
class PrimaryCaps(layers.Layer):

    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=dim_capsule * n_channels,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  activation='relu')
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels
        self.kernel_size = kernel_size   #
        self.strides = strides           #
        self.padding = padding

    def build(self, input_shape):
        # Important: build the internal Conv2D layer using input shape
        self.conv.build(input_shape)
        super().build(input_shape)  # Let Keras know the layer is built


    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = tf.reshape(outputs, (-1, outputs.shape[1] * outputs.shape[2] * self.n_channels, self.dim_capsule))
        return squash(outputs)


    def get_config(self):
        # hook in to keras Layer to modify layer's config on reload
        config = super().get_config()
        config.update({
            "dim_capsule": self.dim_capsule,
            "n_channels": self.n_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config



@register_keras_serializable()
class DigitCaps(layers.Layer):
    # DigitCaps Layer / Higher-level capsules (e.g. detecting objects like digits or lungs)

    def __init__(self, num_capsule, dim_capsule, routing_iters=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
                                        self.input_dim_capsule, self.dim_capsule],
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 2)
        inputs_tiled = tf.expand_dims(inputs_expand, 3)
        inputs_tiled = tf.tile(inputs_tiled, [1, 1, self.num_capsule, 1, 1])
        inputs_hat = tf.matmul(inputs_tiled, self.W)

        b = tf.zeros(shape=[tf.shape(inputs)[0], self.input_num_capsule, self.num_capsule, 1, 1])

        # Dynamic Routing by Agreement algo
        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)  # coupling coefficient, beacause of softmax(...) all c's connected to a single higher capsule sum to 1.
            s = tf.reduce_sum(c * inputs_hat, axis=1, keepdims=True)  # weighted sum along axis=1
            v = squash(s, axis=-2)    # shrinks small vectors to zero and large vectors to unit vectors
            if i < self.routing_iters - 1:
                b += tf.reduce_sum(inputs_hat * v, axis=-1, keepdims=True)

        return tf.squeeze(v, axis=1)


    def get_config(self):
        # hook in to keras Layer to modify layer's config on reload
        config = super().get_config()
        config.update({
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "routing_iters": self.routing_iters
        })
        return config



# Length Layer
@register_keras_serializable()
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))



# Margin Loss for Capsule Networks
def margin_loss(y_true, y_pred):
    # y_true is a one-hot vector
    # y_pred is the Length() output: vector of shape [batch_size, num_classes] (each value ≈ class presence probability)
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lambda_val * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


capsnet_custom_objects = {
    'PrimaryCaps': PrimaryCaps,
    'DigitCaps': DigitCaps,
    'Length': Length,
    'margin_loss': margin_loss
}

    
            
        
        
        
    
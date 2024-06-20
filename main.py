import streamlit as st
import functools
import os
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io

st.title("Neural Style Transfer with TensorFlow")

#st.write("TF Version: ", tf._version_)
#st.write("TF-Hub version: ", hub._version_)

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

@functools.lru_cache(maxsize=None)
def load_image_from_bytes(image_bytes, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images from bytes."""
    img = tf.io.decode_image(image_bytes, channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def show_image(image, title=''):
    plt.imshow(image[0])
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt.gcf())

content_image_file = st.file_uploader('Upload content image', type=['jpg', 'jpeg', 'png'])
style_image_file = st.file_uploader('Upload style image', type=['jpg', 'jpeg', 'png'])
output_image_size = 256

if content_image_file and style_image_file:
    content_image_bytes = content_image_file.read()
    style_image_bytes = style_image_file.read()
    
    content_img_size = (output_image_size, output_image_size)
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image_from_bytes(content_image_bytes, content_img_size)
    style_image = load_image_from_bytes(style_image_bytes, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')


    if st.button('Generate Stylized Image'):
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        hub_module = hub.load(hub_handle)

        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

        st.write("Stylized Image")
        show_image(stylized_image, 'Stylized image')

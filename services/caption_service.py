import os
os.environ["HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["KERAS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras"))
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

import pickle
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, Embedding, LSTM, Bidirectional,
        RepeatVector, Dot, Activation, Lambda, Concatenate
    )
    TF_AVAILABLE = True
except (ImportError, Exception):
    tf = None
    TF_AVAILABLE = False

try:
    import streamlit as st
    cache_resource = st.cache_resource
except ImportError:
    def cache_resource(fn):
        return fn


@cache_resource
def get_feature_extractor():
    """Load MobileNetV2 feature extractor (output shape: 1280)."""
    local_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras", "models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"))
    if os.path.exists(local_weights):
        base_model = MobileNetV2(weights=None)
        base_model.load_weights(local_weights)
    else:
        base_model = MobileNetV2(weights="imagenet")
    feature_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return feature_model


@cache_resource
def get_caption_model(model_path="mymodel.h5"):
    """Load the LSTM attention model with Python-bytecode-safe fallback."""
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        # Reconstruct identical architecture and load trained weights
        inputs1 = Input(shape=(1280,), name="input_3")
        fe1 = Dropout(0.3, name="dropout")(inputs1)
        fe2 = Dense(256, activation="relu", name="dense")(fe1)
        fe2_projected = RepeatVector(34, name="repeat_vector")(fe2)
        fe2_projected = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional")(fe2_projected)

        inputs2 = Input(shape=(34,), name="input_4")
        se1 = Embedding(8768, 256, name="embedding")(inputs2)
        se2 = Dropout(0.3, name="dropout_1")(se1)
        se3 = Bidirectional(LSTM(256, return_sequences=True), name="bidirectional_1")(se2)

        attention = Dot(axes=[2, 2], name="dot")([fe2_projected, se3])
        attention_scores = Activation("softmax", name="activation")(attention)
        attention_context = Lambda(lambda x: tf.einsum("ijk,ijl->ikl", x[0], x[1]), name="lambda")([attention_scores, se3])
        context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1), name="tf.math.reduce_sum")(attention_context)

        decoder_input = Concatenate(axis=-1, name="concatenate")([context_vector, fe2])
        decoder1 = Dense(256, activation="relu", name="dense_1")(decoder_input)
        outputs = Dense(8768, activation="softmax", name="dense_2")(decoder1)

        built_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        built_model.load_weights(model_path)
        return built_model


@cache_resource
def get_tokenizer(tokenizer_path="tokenizer.pkl"):
    """Load serialized tokenizer."""
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def extract_features_from_image(image_input, feature_extractor=None):
    """
    Preprocess image and extract 1280-dim feature vector.
    Accepts PIL Image, file path, or file-like object.
    """
    if feature_extractor is None:
        feature_extractor = get_feature_extractor()

    if isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        # File-like object (e.g., Streamlit UploadedFile)
        pil_img = Image.open(image_input).convert("RGB")

    resized_img = pil_img.resize((224, 224))
    img_array = img_to_array(resized_img)
    img_array = img_array.reshape((1, 224, 224, 3))
    preprocessed_img = preprocess_input(img_array)

    features = feature_extractor.predict(preprocessed_img, verbose=0)
    return features


def get_word_from_index(index, tokenizer):
    """Map tokenizer integer index to word string."""
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)


def generate_base_caption(image_input, max_caption_length=34, model_path="mymodel.h5", tokenizer_path="tokenizer.pkl"):
    """
    Full stage-1 inference: extract features -> autoregressive LSTM decoding -> clean caption.
    """
    feature_extractor = get_feature_extractor()
    caption_model = get_caption_model(model_path)
    tokenizer = get_tokenizer(tokenizer_path)

    image_features = extract_features_from_image(image_input, feature_extractor)

    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        predicted_index = int(np.argmax(yhat))
        predicted_word = get_word_from_index(predicted_index, tokenizer)

        if predicted_word is None or predicted_word == "endseq":
            break

        caption += " " + predicted_word

    # Clean boundary tokens and extra spaces
    clean_caption = caption.replace("startseq", "").replace("endseq", "").strip()
    return clean_caption

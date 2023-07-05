import numpy as np
import tensorflow as tf

from dataclasses import asdict, dataclass, field
from visualime.explain import explain_classification
from visualime.explain import render_explanation
from PIL import Image

IMG_SIZE = 224
model = tf.keras.models.load_model("model")


# EXPLAINER CONFIG -----------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class LIMEConfig:
    """Class for specifying LIME config params"""

    segmentation_method: str = "felzenszwalb"
    num_of_samples: int = 50
    p: float = 0.33


@dataclass(frozen=True, kw_only=True)
class RenderConfig:
    """Class for specifying LIME config params"""

    coverage: float = 0.15
    opacity: float = 0.5
    positive: str = "violet"
    negative: str = None


# FUNCTIONS FOR INFERENCE  ---------------------------------------------------
def preprocess_predict(batch: np.ndarray):
    """
    input: np array of size (batch, IMG_SIZE, IMG_SIZE, 3)
    output: np array of size (batch, IMG_SIZE, IMG_SIZE, 3)
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(batch)


def predict_batch(proc_batch: np.ndarray):
    """
    input: np array of size (batch, IMG_SIZE, IMG_SIZE, 3)
    output: np array of size (batch, 4) where every row represents the softmax
            output of an image inside the given batch
    """
    return model.predict(proc_batch, verbose=0)


def decode_model_output(output: np.ndarray) -> str:
    MODEL_OUTPUT_MAP = ["Tel Aviv", "Jerusalem", "Berlin", "Hamburg"]
    return MODEL_OUTPUT_MAP[int(np.argmax(output, axis=1))]


def predict(image):
    resized_img_arr = image.resize((IMG_SIZE, IMG_SIZE))
    prediction = predict_batch(
        preprocess_predict(np.array(resized_img_arr)[None, :, :, :])
    )
    result = decode_model_output(prediction)
    return result


# FUNCTIONS FOR EXPLANATION  -------------------------------------------------
def preprocess_explain(img):
    return np.array(img.resize((IMG_SIZE, IMG_SIZE)))


def compute_exp(preproc_img, lime_config=LIMEConfig()):
    return explain_classification(
        image=preproc_img,
        predict_fn=lambda batch: predict_batch(preprocess_predict(batch)),
        **asdict(lime_config),
    )


def render_exp(
    preproc_img, segment_mask, segment_weights, render_config=RenderConfig()
):
    return render_explanation(
        preproc_img, segment_mask, segment_weights, **asdict(render_config)
    )


def _explain(img):
    preproc_img = preprocess_explain(img)
    segment_mask, segment_weights = compute_exp(preproc_img)
    return render_exp(preproc_img, segment_mask, segment_weights)


def compute_render_explanation(
    img: Image,
    lime_config: LIMEConfig = LIMEConfig(),
    render_config: RenderConfig = RenderConfig(),
):
    arr_resized_img = preprocess_explain(img)
    segment_mask, segment_weight = explain_classification(
        image=preprocess_predict(arr_resized_img),
        predict_fn=predict_batch,
        **asdict(lime_config),
    )
    return render_explanation(
        arr_resized_img, segment_mask, segment_weight, **asdict(render_config)
    )


#
# def resize_predict(img_arr):
#    downsampled_arr = block_reduce(
#        img_arr, block_size=(2, 2, 1), func=np.mean, cval=np.mean(img_arr)
#    ).astype("uint8")
#    return explainer.predict_image(downsampled_arr)
#
#
# for fn in os.listdir("images_marie"):
#    img = Image.open(f"images_marie/{fn}")
#    pred = resize_predict(np.array(img))
#    print(pred)
#    exp = compute_render_explanation(img)
#    img.resize((IMG_SIZE, IMG_SIZE)).save(f"./images_max/{fn[:-3]}.png")
#    exp.resize((IMG_SIZE, IMG_SIZE)).save(f"./images_max/{fn[:-3]}_pred_{pred}.png")

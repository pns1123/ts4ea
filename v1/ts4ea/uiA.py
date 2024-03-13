import gradio as gr
import io
import json
import numpy as np
import os
import time

from configuration import config2key
from fastapi import FastAPI
from global_config import CONFIG2ID, CONFIG_ENCODER, TOTAL_ROUNDS, SHARE
from PIL import Image
from msg_q import RedisConnection


get_window_url_params = """
    function(right_img, middle_img, left_img, round_counter, personal_params, url_params, pred, left_ref_ind, explanation_id) {
        console.log(right_img, url_params);
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return [right_img, middle_img, left_img, round_counter, personal_params, url_params, pred, left_ref_ind, explanation_id];
        }
    """


set_window_url_params = """
    function(text_input, url_params) {
            const params = new URLSearchParams(window.location.search);
            params.set("text_input", text_input)
            url_params = Object.fromEntries(params);
    	    const queryString = '?' + params.toString();
            // this next line is only needed inside Spaces, so the child frame updates parent
            window.parent.postMessage({ queryString: queryString }, "*")
            return [url_params, text_input];
        }
    """


def deserialize_feature_vec(feature_vec_bytes):
    return np.array(json.loads(feature_vec_bytes.decode())["coef"])


def load_images(y):
    config_id_reference = CONFIG2ID[
        config2key(
            CONFIG_ENCODER.decode(deserialize_feature_vec(y[b"reference_feature_vec"]))
        )
    ]

    config_id_candidate = CONFIG2ID[
        config2key(
            CONFIG_ENCODER.decode(deserialize_feature_vec(y[b"candidate_feature_vec"]))
        )
    ]

    raw_streetview = Image.open(
        os.path.join("streetview", "raw", y[b"filename"].decode())
    )

    exp_reference = Image.open(
        os.path.join(
            "streetview", f"exp_conf_{config_id_reference}", y[b"filename"].decode()
        )
    )

    exp_candidate = Image.open(
        os.path.join(
            "streetview", f"exp_conf_{config_id_candidate}", y[b"filename"].decode()
        )
    )

    return raw_streetview, exp_reference, exp_candidate


def coin_flip():
    return np.random.binomial(n=1, p=0.5) == 1


def compute_reward(ref_left_indicator, choice):
    if choice == SIMILAR_LABEL:
        return 1.0
    elif choice == RIGHT_LABEL and ref_left_indicator:
        return 1.0
    elif choice == LEFT_LABEL and not ref_left_indicator:
        return 1.0
    else:
        return -1.0


def hello(
    img_left,
    img_middle,
    img_right,
    round_counter,
    feature_vec,
    url_params,
    pred,
    left_ref_ind,
    explanation_id,
):
    user_id = url_params.get("user_id", "default_user_id")
    with RedisConnection() as conn:
        conn.xadd(b"hello", {"user_id": url_params.get("user_id", "default_user_id")})

        stream_keys = {str.encode(f"{user_id}_explanations"): b"$"}
        res = conn.xread(stream_keys, count=None, block=0)
        [a, b] = res[0]
        _, y = b[0]

        raw_streetview, exp_reference, exp_candidate = load_images(y)
        img_middle = raw_streetview
        if coin_flip():
            img_left = exp_reference
            img_right = exp_candidate
            left_ref_ind = True
        else:
            img_left = exp_candidate
            img_right = exp_reference
            left_ref_ind = False

        cur_round = y[b"round"].decode()
        pred_city = y[b"pred"].decode()
        explanation_id = y[b"explanation_id"].decode()
        # fix inconsistent labels
        pred = f"The AI predicts that the image was taken in {pred_city}. It provides two explanations for its guess:"

        round_counter = f"Round: {cur_round}/{TOTAL_ROUNDS}. \nDO NOT CLICK NEXT WHILE YOU SEE THIS MESSAGE. Clicking next  before completing all {TOTAL_ROUNDS} rounds will end the current stage prematurely and lead to a loss of your financial compensation!"

        reference_feature_vec = y[b"reference_feature_vec"].decode()

    return [
        img_left,
        img_middle,
        img_right,
        round_counter,
        reference_feature_vec,
        url_params,
        pred,
        left_ref_ind,
        explanation_id,
    ]


def load_new_round(
    img_left,
    img_middle,
    img_right,
    round_counter,
    url_params,
    feature_vec,
    city_choice,
    explanation_choice,
    pred,
    submit,
    warning,
    left_ref_ind,
    explanation_id,
):
    # currently it is not clear why city_choice and explanation_choice
    # are sometimes passed as "" or None when no choice is made
    # REMOVE THIS IMPLICT BOOLEAN CHECK WHEN POSSIBLE
    if not city_choice or not explanation_choice:
        return [
            img_left,
            img_middle,
            img_right,
            round_counter,
            json.dumps(feature_vec),
            url_params,
            city_choice,
            explanation_choice,
            pred,
            submit,
            gr.update(visible=True),
            left_ref_ind,
            explanation_id,
        ]
    else:
        reward = compute_reward(left_ref_ind, explanation_choice)
        user_id = url_params.get("user_id", "default_user_id")
        with RedisConnection() as conn:
            stream_name = str.encode(f"{user_id}_reward_history")
            conn.xadd(
                stream_name,
                {
                    "reward": reward,
                    "candidate_feature_vec": json.dumps(feature_vec),
                    "city_choice": city_choice,
                    "explanation_choice": explanation_choice,
                    "explanation_id": explanation_id,
                },
            )

            stream_name = str.encode(f"{user_id}_explanations")
            stream_keys = {stream_name: b"$"}
            res = conn.xread(stream_keys, count=None, block=0)
            [a, b] = res[0]
            _, y = b[0]

            raw_streetview, exp_reference, exp_candidate = load_images(y)
            img_middle = raw_streetview
            if coin_flip():
                img_left = exp_reference
                img_right = exp_candidate
                left_ref_ind = True
            else:
                img_left = exp_reference
                img_right = exp_candidate
                left_ref_ind = False

            cur_round = y[b"round"].decode()
            pred_city = y[b"pred"].decode()
            explanation_id = y[b"explanation_id"].decode()
            pred = f"The AI predicts that the image was taken in {pred_city}. It gives two explanations for its guess:"

            # feature_vec = y[b"configA"]
            feature_vec = {
                "coef": json.loads(y[b"candidate_feature_vec"].decode())["coef"]
            }

            if int(cur_round) == TOTAL_ROUNDS + 1:
                round_counter = f"You completed all {TOTAL_ROUNDS} rounds. \nPlease click Next to complete the final stage."
                return [
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    round_counter,
                    feature_vec,
                    url_params,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    left_ref_ind,
                    explanation_id,
                ]

            else:
                round_counter = f"Round: {cur_round}/{TOTAL_ROUNDS}. \nDO NOT CLICK NEXT before completing all {TOTAL_ROUNDS} rounds. This will end the current stage prematurely and lead to a loss of your financial compensation!"

                return [
                    img_left,
                    img_middle,
                    img_right,
                    round_counter,
                    feature_vec,
                    url_params,
                    gr.update(value=None),
                    gr.update(value=None),
                    pred,
                    submit,
                    gr.update(visible=False),
                    left_ref_ind,
                    explanation_id,
                ]


with gr.Blocks() as interface:
    url_params = gr.JSON(value={}, visible=False, label="URL Params")
    # ref_img_indicator == True <=> ref_exp displayed on left
    left_ref_img_indicator = gr.Checkbox(visible=False)

    pred = gr.Text("", visible=True, label="AI Prediction", interactive=False)
    with gr.Row():
        left_img = gr.Image(interactive=False, label="Explanation 1")
        raw_img = gr.Image(interactive=False, label="Streetview Image")
        right_img = gr.Image(interactive=False, label="Explanation 2")

    with gr.Row():
        city_choice = gr.Radio(
            choices=["Berlin", "Hamburg", "Jerusalem", "Tel Aviv"],
            label="1) Select the city this Streetview photo was taken in:",
            value=None,
        )

    with gr.Row():
        LEFT_LABEL = "Explanation 1 (left)"
        RIGHT_LABEL = "Explanation 2 (right)"
        SIMILAR_LABEL = "There is no difference"
        explanation_choice = gr.Radio(
            choices=[
                LEFT_LABEL,
                SIMILAR_LABEL,
                RIGHT_LABEL,
            ],
            label="2) Select the explanation you find more helpful in making your decision.",
            value=None,
        )

    with gr.Row():
        round_counter = gr.Textbox(value="", label="Information", interactive=False)

    with gr.Row():
        feature_vec = gr.JSON(value={}, label="DEV INFO", visible=False)
        explanation_id = gr.Textbox(value="", visible=False)

    with gr.Row():
        warning = gr.Textbox(
            "You must choose a city and an explanation before you can continue.",
            visible=False,
            label="ERROR",
            interactive=False,
        )

    with gr.Row():
        submit = gr.Button("Continue")

    interface.load(
        _js=get_window_url_params,
        fn=hello,
        inputs=[
            left_img,
            raw_img,
            right_img,
            round_counter,
            feature_vec,
            url_params,
            pred,
            left_ref_img_indicator,
            explanation_id,
        ],
        outputs=[
            left_img,
            raw_img,
            right_img,
            round_counter,
            feature_vec,
            url_params,
            pred,
            left_ref_img_indicator,
            explanation_id,
        ],
    )
    submit.click(
        fn=load_new_round,
        inputs=[
            left_img,
            raw_img,
            right_img,
            round_counter,
            url_params,
            feature_vec,
            city_choice,
            explanation_choice,
            pred,
            submit,
            warning,
            left_ref_img_indicator,
            explanation_id,
        ],
        outputs=[
            left_img,
            raw_img,
            right_img,
            round_counter,
            feature_vec,
            url_params,
            city_choice,
            explanation_choice,
            pred,
            submit,
            warning,
            left_ref_img_indicator,
            explanation_id,
        ],
    )


interface.launch(debug=True, server_name="0.0.0.0", share=SHARE)

import gradio as gr
import io
import json
import numpy as np
import os
import time

from fastapi import FastAPI
from global_config import N_ROUNDS, SHARE
from PIL import Image
from msg_q import RedisConnection

get_window_url_params = """
    function(right_img, middle_img, left_img, round_counter, personal_params, url_params, pred, left_ref_ind) {
        console.log(right_img, url_params);
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return [right_img, middle_img, left_img, round_counter, personal_params, url_params, pred, left_ref_ind];
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


def coin_flip():
    return np.random.binomial(n=1, p=0.5) == 1


def compute_reward(ref_left_indicator, choice):
    print(ref_left_indicator, type(ref_left_indicator), choice, type(choice))
    if choice == SIMILAR_LABEL:
        return 1.0
    elif choice == RIGHT_LABEL and ref_left_indicator:
        return 1.0
    elif choice == LEFT_LABEL and not ref_left_indicator:
        return 1.0
    else:
        return -1.0

def proc_pred_city(pred_city):
    if pred_city == "Westjerusalem":
        return "Jerusalem"
    elif pred_city == "Tel_Aviv":
        return "Tel Aviv"
    else: 
        return pred_city


def hello(
    img_left,
    img_middle,
    img_right,
    round_counter,
    feature_vec,
    url_params,
    pred,
    left_ref_ind,
):

    user_id = url_params.get("user_id", "default_user_id")
    with RedisConnection() as conn:
        conn.xadd(b"hello", {"user_id": url_params.get("user_id", "default_user_id")})

        stream_keys = {str.encode(f"{user_id}_explanations"): b"$"}
        # stream_keys = {b"test_stream": b"$"}
        print(f"hello listening on {stream_keys}")
        res = conn.xread(stream_keys, count=None, block=0)
        [a, b] = res[0]
        _, y = b[0]
        img_middle = Image.open(io.BytesIO(y[b"img_bytes"]))

        if coin_flip():
            print("side A")
            img_left = Image.open(io.BytesIO(y[b"ref_exp_bytes"]))
            img_right = Image.open(io.BytesIO(y[b"exp_adjusted_bytes"]))
            left_ref_ind = True
        else:
            print("side B")
            img_left = Image.open(io.BytesIO(y[b"exp_adjusted_bytes"]))
            img_right = Image.open(io.BytesIO(y[b"ref_exp_bytes"]))
            left_ref_ind = False

        cur_round = y[b"round"].decode()
        pred_city = proc_pred_city(y[b"pred"].decode())
        # fix inconsistent labels
        pred = f"The AI predicts that the image was taken in {pred_city}. It gives two explanations for its guess:"

        feature_vec = y[b"feature_vec"]
        round_counter = f"Round: {cur_round}/{N_ROUNDS}. \nDO NOT CLICK NEXT WHILE YOU SEE THIS MESSAGE. Clicking next  before completing all {N_ROUNDS} rounds will end the current stage prematurely and lead to a loss of your financial compensation!"

    return [
        img_left,
        img_middle,
        img_right,
        round_counter,
        feature_vec,
        url_params,
        pred,
        left_ref_ind,
    ]


def load_images(
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
):
    print(city_choice, type(city_choice), left_ref_ind, type(left_ref_ind))
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
        ]
    else:
        reward = compute_reward(left_ref_ind, explanation_choice)
        print("reward:", reward)
        user_id = url_params.get("user_id", "default_user_id")
        with RedisConnection() as conn:
            stream_name = str.encode(f"{user_id}_reward_history")
            print("reward:", reward)
            print("feature_vec:", feature_vec)
            conn.xadd(
                stream_name,
                {
                    "reward": reward,
                    "feature_vec": feature_vec,
                    "city_choice": city_choice,
                    "explanation_choice": explanation_choice,
                },
            )

            stream_name = str.encode(f"{user_id}_explanations")
            stream_keys = {stream_name: b"$"}
            print(f"load_images xread on {stream_keys}")
            res = conn.xread(stream_keys, count=None, block=0)
            [a, b] = res[0]
            _, y = b[0]

            img_middle = Image.open(io.BytesIO(y[b"img_bytes"]))
            if coin_flip():
                img_left = Image.open(io.BytesIO(y[b"ref_exp_bytes"]))
                img_right = Image.open(io.BytesIO(y[b"exp_adjusted_bytes"]))
                left_ref_ind = True
            else:
                img_left = Image.open(io.BytesIO(y[b"exp_adjusted_bytes"]))
                img_right = Image.open(io.BytesIO(y[b"ref_exp_bytes"]))
                left_ref_ind = False

            cur_round = y[b"round"].decode()
            pred_city = proc_pred_city(y[b"pred"].decode())
            pred = f"The AI predicts that the image was taken in {pred_city}. It gives two explanations for its guess:"
            feature_vec = y[b"feature_vec"]

            if int(cur_round) == N_ROUNDS + 1:
                round_counter = f"You completed all {N_ROUNDS} rounds. \nPlease click Next to complete the final stage."
                return [
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    round_counter,
                    feature_vec,
                    url_params,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    pred,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    left_ref_ind,
                ]

            else:
                round_counter = f"Round: {cur_round}/{N_ROUNDS}. \nDO NOT CLICK NEXT before completing all {N_ROUNDS} rounds. This will end the current stage prematurely and lead to a loss of your financial compensation!"

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
            choices=["Berlin", "Hamburg", "Jerusalem", "Tel-Aviv"],
            label="1) Select the city this Streetview photo was taken in:",
            value=None,
        )

    with gr.Row():
        LEFT_LABEL = "Explanation 1 (left)"
        RIGHT_LABEL = "Explanation 2 (right)"
        SIMILAR_LABEL = "I cannot spot a difference"
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
        round_counter = gr.Textbox(value=f"", label="Information", interactive=False)

    with gr.Row():
        feature_vec = gr.JSON(value={}, label="feature_vec encoded", visible=False)

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
        ],
    )
    submit.click(
        fn=load_images,
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
        ],
    )


interface.launch(debug=True, server_name="0.0.0.0", share=SHARE)

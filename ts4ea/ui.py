import gradio as gr
import io
import json
import numpy as np
import os
import time

from fastapi import FastAPI
from global_config import N_ROUNDS
from PIL import Image
from msg_q import RedisConnection

get_window_url_params = """
    function(right_img, middle_img, left_img, round_counter, personal_params, url_params, pred) {
        console.log(right_img, url_params);
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return [right_img, middle_img, left_img, round_counter, personal_params, url_params, pred];
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


def hello(
    img_left, img_middle, img_right, round_counter, feature_vec, url_params, pred
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
        img_left = Image.open(io.BytesIO(y[b"ref_exp_bytes"]))
        img_right = Image.open(io.BytesIO(y[b"exp_adjusted_bytes"]))
        cur_round = y[b"round"].decode()
        pred = y[b"pred"]

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
    ]


def load_images(
    url_params, feature_vec, reward, city_choice, explanation_choice, pred, submit
):
    print(f"load_images: url_params = {url_params}")
    user_id = url_params.get("user_id", "default_user_id")
    with RedisConnection() as conn:
        stream_name = str.encode(f"{user_id}_reward_history")
        print(f"load_images xadd on {stream_name}")
        print("city_choice = ", city_choice)
        print("explanation_choice = ", explanation_choice)
        conn.xadd(
            stream_name,
            {
                "reward": reward,
                "feature_vec": str.encode(feature_vec),
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
        img_left = Image.open(io.BytesIO(y[b"ref_exp_bytes"]))
        img_right = Image.open(io.BytesIO(y[b"exp_adjusted_bytes"]))
        cur_round = y[b"round"].decode()
        pred = y[b"pred"]
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
                submit
            ]


with gr.Blocks() as interface:
    url_params = gr.JSON(visible=False, label="URL Params")
    pred = gr.Text("", visible=False)
    # text_input = gr.Text(label="Input")
    # text_output = gr.Text(label="Output")

    with gr.Row():
        raw_img = gr.Image(interactive=False, label="Streetview Image")
        ref_img = gr.Image(interactive=False, label="Explanation 1")
        adj_img = gr.Image(interactive=False, label="Explanation 2")

    with gr.Row():
        city_choice = gr.Radio(
            choices=["Berlin", "Hamburg", "Jerusalem", "Tel-Aviv"],
            label="1) Select the city this Streetview photo was taken in:",
        )

    with gr.Row():
        REFERENCE_LABEL = "Explanation 1 (Left)"
        ADJUSTED_LABEL = "Explanation 2 (Right)"
        CONTROL_LABEL = "Explanations are Equal"
        explanation_label2reward = {
            REFERENCE_LABEL: 0.0,
            ADJUSTED_LABEL: 1.0,
            #CONTROL_LABEL: 1.0,
        }
        explanation_choice = gr.Radio(
            choices=[
                #CONTROL_LABEL,
                REFERENCE_LABEL,
                ADJUSTED_LABEL,
            ],
            label="2) Select the explanation you find more helpful in identifying the location of the photo.",
        )

    with gr.Row():
        round_counter = gr.Textbox(value=f"", label="Information")

    with gr.Row():
        feature_vec = gr.JSON(value={}, label="feature_vec encoded", visible=False)

    with gr.Row():
        submit = gr.Button("Continue")

    interface.load(
        _js=get_window_url_params,
        fn=hello,
        inputs=[
            ref_img,
            raw_img,
            adj_img,
            round_counter,
            feature_vec,
            url_params,
            pred,
        ],
        outputs=[
            ref_img,
            raw_img,
            adj_img,
            round_counter,
            feature_vec,
            url_params,
            pred,
        ],
    )

    submit.click(
        fn=load_images,
        inputs=[
            url_params,
            feature_vec,
            gr.Number(value=explanation_label2reward.get(city_choice), visible=False),
            city_choice,
            explanation_choice,
            pred,
            submit,
        ],
        outputs=[
            ref_img,
            raw_img,
            adj_img,
            round_counter,
            feature_vec,
            url_params,
            city_choice,
            explanation_choice,
            pred,
            submit,
        ],
    )


interface.launch(debug=True, server_name="0.0.0.0", share=True)

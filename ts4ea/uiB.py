import gradio as gr
import io
import json
import numpy as np
import os
import time

from configuration import config2key, LIMEConfig, RenderConfig
from dataclasses import asdict
from fastapi import FastAPI
from global_config import CONFIG2ID, CONFIG_ENCODER, TOTAL_ROUNDS, SHARE
from PIL import Image
from msg_q import RedisConnection


get_window_url_params = """
    function(right_img, middle_img, left_img, city_choice, round_counter, personal_params, url_params, pred, left_ref_ind, explanation_id, filename, model_pred) {
        console.log(right_img, url_params);
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return [right_img, middle_img, left_img, city_choice, round_counter, personal_params, url_params, pred, left_ref_ind, explanation_id, filename, model_pred];
        }
    """


def pred_text(model_pred):
    return f"1) The AI predicts that the image was taken in {model_pred.upper()}. What is your assessment?"


def model_pred_correct(filename_label, pred_label):
    match pred_label:
        case "Berlin":
            return filename_label == "BERLIN"
        case "Hamburg":
            return filename_label == "HAMBURG"
        case "Tel Aviv":
            return filename_label == "TELAVIV"
        case "Jerusalem":
            return filename_label == "WESTJERUSALEM"


def valid_label(filename_label):
    match filename_label:
        case "BERLIN":
            return "Berlin"
        case "HAMBURG":
            return "Hamburg"
        case "TELAVIV":
            return "Tel Aviv"
        case "WESTJERUSALEM":
            return "Jerusalem"


def deserialize_feature_vec(feature_vec_bytes):
    return np.array(json.loads(feature_vec_bytes.decode())["coef"])


def load_images(y):
    config_reference = CONFIG_ENCODER.encode(
        {**asdict(LIMEConfig()), **asdict(RenderConfig())}
    )
    config_id_reference = CONFIG2ID[config2key(CONFIG_ENCODER.decode(config_reference))]

    config_adjusted = deserialize_feature_vec(y[b"candidate_feature_vec"])
    config_id_adjusted = CONFIG2ID[config2key(CONFIG_ENCODER.decode(config_adjusted))]

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
            "streetview", f"exp_conf_{config_id_adjusted}", y[b"filename"].decode()
        )
    )

    return (
        raw_streetview,
        exp_reference,
        exp_candidate,
        config_reference,
        config_adjusted,
    )


def coin_flip():
    return np.random.binomial(n=1, p=0.5) == 1


def compute_reward(left_ref_indicator, choice):
    if choice == STRONGLY_DISAGREE or choice == DISAGREE:
        return -1.0
    elif choice == STRONGLY_AGREE or choice == AGREE:
        return 1.0
    else:
        raise ValueError("Invalid choice: neither yes or no")


def hello(
    img_left,
    img_middle,
    img_right,
    city_choice,
    round_counter,
    feature_vec,
    url_params,
    pred,
    left_ref_ind,
    explanation_id,
    filename,
    model_pred,
):
    user_id = url_params.get("user_id", "default_user_id")
    user_group = url_params.get("user_group", "treatment")

    with RedisConnection() as conn:
        conn.xadd(b"hello", {"user_id": user_id})

        stream_keys = {str.encode(f"{user_id}_explanations"): b"$"}
        res = conn.xread(stream_keys, count=None, block=0)
        [a, b] = res[0]
        _, y = b[0]

        (
            raw_streetview,
            exp_reference,
            exp_candidate,
            config_reference,
            config_adjusted,
        ) = load_images(y)
        img_middle = raw_streetview
        if user_group == "control":
            feature_vec = {"coef": config_reference}
            img_right = exp_reference
        else:
            feature_vec = {"coef": config_adjusted}
            img_right = exp_candidate

        #print("user_group", user_group, "feature_vec", feature_vec, "mu",  json.loads(y[b"model_params"])["mu"])
        cur_round = y[b"round"].decode()
        explanation_id = y[b"explanation_id"].decode()
        filename = y[b"filename"].decode()
        model_pred = y[b"pred"].decode()
        # fix inconsistent labels
        pred = pred_text(model_pred)

        round_counter = f"Round: {cur_round}/{TOTAL_ROUNDS}. \nDO NOT CLICK NEXT WHILE YOU SEE THIS MESSAGE. Clicking next  before completing all {TOTAL_ROUNDS} rounds will end the current stage prematurely and lead to a loss of your financial compensation!"

        reference_feature_vec = y[b"reference_feature_vec"].decode()

    return [
        img_left,
        img_middle,
        img_right,
        gr.update(label=pred),
        round_counter,
        reference_feature_vec,
        url_params,
        pred,
        left_ref_ind,
        explanation_id,
        filename,
        model_pred,
    ]


def show_result(
    city_choice,
    explanation_choice,
    submit,
    check,
    warning,
    filename,
    model_pred,
    feedback,
):
    # currently it is not clear why city_choice and explanation_choice
    # are sometimes passed as "" or None when no choice is made
    # REMOVE THIS IMPLICT BOOLEAN CHECK WHEN POSSIBLE
    if not city_choice or not explanation_choice:
        return [
            city_choice,
            explanation_choice,
            submit,
            check,
            gr.update(visible=True),
            filename,
            model_pred,
            feedback, 
        ]
    else:
        
        filename_label = filename.split("_")[0]
        user_decision_correct = model_pred_correct(filename_label, model_pred) == (city_choice==CORRECT_CHOICE)
        user_decision_str = "CORRECT" if user_decision_correct else "WRONG"
        feedback_text = f"Your decision was {user_decision_str}. The AI predicted {model_pred.upper()} and the image was taken in {valid_label(filename_label).upper()}."

        return [
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            filename,
            model_pred,
            gr.update(value=feedback_text,
                      visible=True,)
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
    check,
    filename,
    model_pred,
    feedback,
):
    # currently it is not clear why city_choice and explanation_choice
    # are sometimes passed as "" or None when no choice is made
    # REMOVE THIS IMPLICT BOOLEAN CHECK WHEN POSSIBLE
    reward = compute_reward(left_ref_ind, explanation_choice)
    user_id = url_params.get("user_id", "default_user_id")
    user_group = url_params.get("user_group", "treatment")
    with RedisConnection() as conn:
        stream_name = str.encode(f"{user_id}_reward_history")
        conn.xadd(
            stream_name,
            {
                "reward": reward,
                "candidate_feature_vec": json.dumps(feature_vec),
                "user_group": user_group,
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

        (
            raw_streetview,
            exp_reference,
            exp_candidate,
            config_reference,
            config_adjusted,
        ) = load_images(y)
        img_middle = raw_streetview
        if user_group == "control":
            feature_vec = {"coef": list(config_reference)}
            img_right = exp_reference
        else:
            feature_vec = {"coef": list(config_adjusted)}
            img_right = exp_candidate
        #print("user_group", user_group, "feature_vec", feature_vec, "mu",  json.loads(y[b"model_params"])["mu"])

        cur_round = y[b"round"].decode()
        explanation_id = y[b"explanation_id"].decode()
        filename = y[b"filename"].decode()
        model_pred = y[b"pred"].decode()
        pred = pred_text(model_pred)

        # feature_vec = {
        #    "coef": json.loads(y[b"candidate_feature_vec"].decode())["coef"]
        # }
        # print(feature_vec, feature_vec_)

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
                gr.update(visible=False),
                filename,
                model_pred,
                gr.update(visible=False),
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
                gr.update(
                    value=None,
                    interactive=True,
                    label=pred,
                ),
                gr.update(value=None, interactive=True),
                pred,
                gr.update(visible=False),
                gr.update(visible=False),
                left_ref_ind,
                explanation_id,
                gr.update(visible=True),
                filename,
                model_pred,
                gr.update(visible=False),
            ]

css = """
#warning {
    background-color: #DC143C;
}
#feedback {
    background-color: #000080;
}
"""

with gr.Blocks(css=css) as interface:
    url_params = gr.JSON(value={}, visible=False, label="URL Params")
    left_ref_img_indicator = gr.Checkbox(visible=False)

    pred = gr.Text("", visible=False, label="AI Prediction", interactive=False)
    with gr.Row():
        left_img = gr.Image(interactive=False, label="Streetview Image", visible=False)
        raw_img = gr.Image(interactive=False, label="Streetview Image")
        right_img = gr.Image(interactive=False, label="Explanation")

    with gr.Row():
        CORRECT_CHOICE = "the AI's PREDICTION is CORRECT"
        INCORRECT_CHOICE = "the AI's PREDICTION is INCORRECT"
        city_choice = gr.Radio(
            choices=[CORRECT_CHOICE, INCORRECT_CHOICE],
            label=None,
            value=None,
        )

    with gr.Row():
        STRONGLY_DISAGREE = "STRONGLY DISAGREE"
        DISAGREE = "DISAGREE"
        AGREE = "AGREE"
        STRONGLY_AGREE = "STRONGLY AGREE"
        explanation_choice = gr.Radio(
            choices=[
                STRONGLY_DISAGREE,
                DISAGREE,
                AGREE,
                STRONGLY_AGREE,
            ],
            label="2) Keep in mind that explanations are adjusted based on your feedback. Please rate: THE EXPLANATION IS HELPFUL IN ASSESSING IF THE AI'S DECISION IS CORRECT.",
            value=None,
        )

    with gr.Row():
        feedback = gr.Textbox(value="", label="Feedback", interactive=False, visible=False, elem_id="feedback")

    with gr.Row():
        round_counter = gr.Textbox(value="", label="Information", interactive=False)

    with gr.Row():
        feature_vec = gr.JSON(value={}, label="DEV INFO", visible=False)
        explanation_id = gr.Textbox(value="", visible=False)
        filename = gr.Textbox(value="", visible=False)
        model_pred = gr.Textbox(value="", visible=False)

    with gr.Row():
        warning = gr.Textbox(
            "You must choose a city and an explanation before you can continue.",
            visible=False,
            label="ERROR",
            interactive=False,
            elem_id="warning",
        )

    with gr.Row():
        check = gr.Button("Submit & Check Answer")
        submit = gr.Button("Continue", visible=False)

    interface.load(
        _js=get_window_url_params,
        fn=hello,
        inputs=[
            left_img,
            raw_img,
            right_img,
            city_choice,
            round_counter,
            feature_vec,
            url_params,
            pred,
            left_ref_img_indicator,
            explanation_id,
            filename,
            model_pred,
        ],
        outputs=[
            left_img,
            raw_img,
            right_img,
            city_choice,
            round_counter,
            feature_vec,
            url_params,
            pred,
            left_ref_img_indicator,
            explanation_id,
            filename,
            model_pred,
        ],
    )

    check.click(
        fn=show_result,
        inputs=[
            city_choice,
            explanation_choice,
            submit,
            check,
            warning,
            filename,
            model_pred,
            feedback,
        ],
        outputs=[
            city_choice,
            explanation_choice,
            submit,
            check,
            warning,
            filename,
            model_pred,
            feedback,
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
            check,
            filename,
            model_pred,
            feedback,
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
            check,
            filename,
            model_pred,
            feedback,
        ],
    )


interface.launch(debug=True, server_name="0.0.0.0", share=True)#SHARE)

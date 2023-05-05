import openai

import random
import json
import time
import os

from grit_model import DenseCaptioning

openai.api_key = os.getenv("OPENAI_API_KEY") 

def prepare_chatgpt_message(caption, dense_caption, vids):
    texts = []
    for cap, dcap, vid in zip(caption, dense_caption, vids):
        text = f"""
            video_id:
            {vid}
            action_caption:
            {cap}
            frame_1_dense_caption:
            {dcap[0]}
            frame_2_dense_caption:
            {dcap[1]}
            frame_3_dense_caption:
            {dcap[2]}
            frame_4_dense_caption:
            {dcap[3]}
        """
        texts.append(text)

    prompt = f"""
        Given a short caption describing a driving action about the ego view car, dense caption data containing object descriptions and positions from multiple frames, \
        create a detailed and coherent video caption that describes the scene and driving behavior without explicitly mentioning individual frames.
        Your description should include the object, color, and position of each element in the scene without using coordinates or numbers.
        Write a single paragraph containing no more than seven sentences.
        Make sure to describe the position of each object using nouns rather than coordinates.

        Steps to complete the task:
        1. Analyze the short caption to understand the driving action taking place in the video.
        2. Extract the object descriptions and positions from the dense caption data of each frame and understand the interaction between different objects.
        3. Observe the changes in object positions and interactions across multiple frames to deduce driving behavior.
        4. Convert the coordinate information to descriptive nouns indicating the position of each object within the scene.
        5. Incorporate the driving action from the short caption and driving behavior inferred from the frames into the description.
        6. Ensure the description does not exceed seven sentences and does not contain any numbers or coordinates.
        7. Review the generated caption for clarity and adherence to the specified rules.

        Provide them in JSON format with the following keys: 
        video_id, final_caption.
    """
    messages = [{"role": "system", "content": prompt}]

    for txt in texts:
        messages.append({'role': 'user', 'content': txt})
    with open('messages.json', 'w') as f:
        json.dump(messages, f, indent=4, sort_keys=True)
    return messages


def call_chatgpt(chatgpt_messages, max_tokens=2048, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model, messages=chatgpt_messages, temperature=1.0, max_tokens=max_tokens
    )
    reply = response["choices"][0]["message"]["content"]
    total_tokens = response["usage"]["total_tokens"]
    return reply, total_tokens


# read caption
with open("video_car/filter_cap.json", "r") as f:
    json_file = json.load(f)
captions = []
video_ids = []

gap = 4
for i, ann in enumerate(json_file["annotations"]):
    if i % gap != 0: continue
    captions.append(ann["caption"])
    video_ids.append(ann["event_path"])

# init grit vit_b
device = 'cuda'
dense_captioning = DenseCaptioning(device)

lim_video_num = 1
pack_dense_caps = []
pack_caps = []
pack_vids = []

iter_id = 0
for caption, video_id in zip(captions[iter_id*lim_video_num:], video_ids[iter_id*lim_video_num:]):
    if os.path.exists(f"./all_json/{str(iter_id).zfill(4)}.json"):
        iter_id += 1
        continue
    # random select a frame from cur video
    # rd_frame = random.randint(0, 15)
    video_cap = []
    frame_ids = [0, 4, 8, 12]
    for rd_frame in frame_ids:
        image_src = f"video_car/image/{video_id}/{str(rd_frame).zfill(2)}.jpg"
        dense_caption = dense_captioning.image_dense_caption(image_src)
        video_cap.append(dense_caption)

    pack_caps.append(caption)
    pack_dense_caps.append(video_cap)
    pack_vids.append(video_id)

    if len(pack_caps) == lim_video_num:
        chatgpt_messages = prepare_chatgpt_message(
            pack_caps, pack_dense_caps, pack_vids
        )
        response, n_tokens = call_chatgpt(chatgpt_messages)
        print(n_tokens)
        data = json.loads("[" + response.replace("\n\n", ",") + "]")
        for i in range(len(data)):
            data[i]["video_id"] = pack_vids[i]
        # Save the results to a JSON file
        with open(f"./all_json/{str(iter_id).zfill(4)}.json", "w") as outfile:
            json.dump(data, outfile)

        pack_dense_caps.clear()
        pack_caps.clear()
        pack_vids.clear()
        iter_id += 1

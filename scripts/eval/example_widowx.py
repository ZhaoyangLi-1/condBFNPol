# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt

'''
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus
import time

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# import ipdb; ipdb.set_trace()
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
# ).to("cuda:0")

# Initialize WidowXClient
robot_ip = "localhost"  # Replace with actual IP if different
robot_port = 5556  # Replace with actual port if different
widowx_client = WidowXClient(host=robot_ip, port=robot_port)

def get_from_camera():
    # Implement the function to capture an image from the Wi`dowX robot's camera
    obs = widowx_client.get_observation()
    image = obs["image_primary"]  # or another key depending on your setup
    return Image.fromarray(image)

def send_robot_action(action):
    # Convert the action into the format expected by the WidowX robot
    # This function needs to be implemented based on how actions are sent to the robot
    # Example placeholder
    widowx_client.act(action)

def main():
    # Capture image from the robot's camera
    image: Image.Image = get_from_camera()
    
    # Format the prompt with the specific goal instruction
    goal_instruction = "move the gripper to the target position"  # Adjust this based on your goal
    prompt = f"In: What action should the robot take to {goal_instruction}?\nOut:"

    # Predict action
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    # Execute the action
    send_robot_action(action)

if __name__ == "__main__":
    main()
'''

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np
import time

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
import  os

import sys
# Append the directory path to sys.path
sys.path.append('../../../octo/examples/')
from envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus


np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=False
)
# image: Image.Image = get_from_camera()
    
# custom to bridge_data_robot
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer("im_size", None, "Image size", required=False)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer(
    "action_horizon", 4, "Length of action sequence to execute/ensemble"
)

# Define the 'deterministic' flag
# flags.DEFINE_boolean('deterministic', default=False, help='Use deterministic mode')

# flags.DEFINE_boolean('temperature', default=False, help='Use temperature mode')

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def main(_):
    # set up the widowx client
    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["start_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=256)
    env = WidowXGym(
        widowx_client, 256, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )
    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

    # load models
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    # import ipdb; ipdb.set_trace()
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    goal_image = jnp.zeros((256, 256, 3), dtype=np.uint8)
    # import ipdb; ipdb.set_trace()
    # goal sampling loop
    while True:
        # reset env
        obs, _ = env.reset()
        # image: Image.Image = get_from_camera()
        image = obs["image_primary"][-1]
    
        if click.confirm("Take a new instruction?", default=True):
            text = input("Instruction?")
        # For logging purposes
        goal_instruction = text
        goal_image = jnp.zeros_like(goal_image) # blank technically

        # Predict action
        # inputs = processor(goal_instruction, goal_image).to("cuda:0", dtype=torch.bfloat16)
        inputs = processor(goal_instruction, Image.fromarray(image)).to(dtype=torch.bfloat16)

        input("Press [Enter] to start.")

        # time.sleep(2.0)

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            print(f"Iteration {t}")
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                # save images
                images.append(obs["image_primary"][-1])
                goals.append(goal_image) # technically not useful

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                with torch.no_grad():
                    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)

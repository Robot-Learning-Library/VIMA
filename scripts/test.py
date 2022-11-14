from vima_bench import make
from vima_bench.tasks.components.encyclopedia import ObjPedia
from vima.utils import *
from vima_bench import *
from example import *
# from vima_bench.tasks.components.encyclopedia import ObjPedia


# env = make(task_name="visual_manipulation", task_kwargs=task_kwargs)
# # env = make(task_name="scene_understanding", task_kwargs=task_kwargs)

# obs = env.reset()
# env.render()
# prompt, prompt_assets = env.prompt, env.prompt_assets
# print(prompt, prompt_assets)

task_name = ['visual_manipulation', 'rotate'][1]

task_kwargs = {
    'rotate': {
    'dragged_obj_express_types': "name",
    'possible_dragged_obj': 'L-shaped block',
    'possible_dragged_obj_texture': ['red'],  # at least two (two objects), color or TextureEntry
    'possible_distractor_obj_texture': ['green'], 
     },

    'visual_manipulation': {
    'dragged_obj_express_types': "name",
    'base_obj_express_types': 'name',
    'possible_dragged_obj': 'block',
    # 'possible_base_obj': 'block',
    'possible_base_obj': 'container',
    'possible_dragged_obj_texture': ['red and purple polka dot'], # at least one of dragged and base needs to be > 1, to give distractor obj a different color
    'possible_base_obj_texture': ['green swirl'],
    'possible_distractor_obj_texture': ['blue'],
    'num_dragged_obj': 3,
    }
    }

test_prompt = {
    'rotate': 'Rotate the red L-shaped block 30 degrees.',
    # 'visual_manipulation': "Put the red and purple polka dot block in the green swirl block."
    'visual_manipulation': "Put the red and purple polka dot block in the green swirl container."
}



model_ckpt = '../models/200M.ckpt'
device = 'cpu'

seed = 42
policy = create_policy_from_ckpt(model_ckpt, device)
env = TimeLimitWrapper(
    ResetFaultToleranceWrapper(
        make(
            task_name=task_name,
            modalities=["segm", "rgb"],
            task_kwargs=task_kwargs[task_name],
            # task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
            seed=seed,
            render_prompt=True,
            display_debug_window=True,
            hide_arm_rgb=False,
            )
    ),
    bonus_steps=2,
)

while True:
    env.global_seed = seed
    # prompt = None
    prompt = test_prompt[task_name]
    obs = env.reset(prompt)
    # obs = env.reset()
    # env.set_prompt_and_assets(prompt=prompt, assets={})  # this will not fully specify the promt in task
    env.render()

    meta_info = env.meta_info
    prompt = env.prompt
    prompt_assets = env.prompt_assets
    elapsed_steps = 0
    inference_cache = {}
    print('next round: ', prompt, prompt_assets.keys())
    while True:
        if elapsed_steps == 0:
            prompt_token_type, word_batch, image_batch = prepare_prompt(
                prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
            )
            word_batch = word_batch.to(device)
            if len(image_batch) > 0:
                image_batch = image_batch.to_torch_tensor(device=device)
            prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                (prompt_token_type, word_batch, image_batch)
            )

            inference_cache["obs_tokens"] = []
            inference_cache["obs_masks"] = []
            inference_cache["action_tokens"] = []
        obs["ee"] = np.asarray(obs["ee"])
        obs = add_batch_dim(obs)
        obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
            device=device
        )
        obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
        obs_token_this_step = obs_token_this_step.squeeze(0)
        obs_mask_this_step = obs_mask_this_step.squeeze(0)
        inference_cache["obs_tokens"].append(obs_token_this_step[0])
        inference_cache["obs_masks"].append(obs_mask_this_step[0])
        max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
        obs_tokens_to_forward, obs_masks_to_forward = [], []
        obs_tokens_this_env, obs_masks_this_env = [], []
        for idx in range(len(inference_cache["obs_tokens"])):
            obs_this_env_this_step = inference_cache["obs_tokens"][idx]
            obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
            required_pad = max_objs - obs_this_env_this_step.shape[0]
            obs_tokens_this_env.append(
                any_concat(
                    [
                        obs_this_env_this_step,
                        torch.zeros(
                            required_pad,
                            obs_this_env_this_step.shape[1],
                            device=device,
                            dtype=obs_this_env_this_step.dtype,
                        ),
                    ],
                    dim=0,
                )
            )
            obs_masks_this_env.append(
                any_concat(
                    [
                        obs_mask_this_env_this_step,
                        torch.zeros(
                            required_pad,
                            device=device,
                            dtype=obs_mask_this_env_this_step.dtype,
                        ),
                    ],
                    dim=0,
                )
            )
        obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
        obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
        obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
        obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
        obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
        obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

        if elapsed_steps == 0:
            action_tokens_to_forward = None
        else:
            action_tokens_to_forward = any_stack(
                [any_stack(inference_cache["action_tokens"], dim=0)],
                dim=0,
            )
            action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
        predicted_action_tokens = policy.forward(
            obs_token=obs_tokens_to_forward,
            action_token=action_tokens_to_forward,
            prompt_token=prompt_tokens,
            prompt_token_mask=prompt_masks,
            obs_mask=obs_masks_to_forward,
        )  # (L, B, E)
        predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(
            0
        )  # (1, B, E)
        dist_dict = policy.forward_action_decoder(predicted_action_tokens)
        actions = {k: v.mode() for k, v in dist_dict.items()}
        action_tokens = policy.forward_action_token(actions)  # (1, B, E)
        action_tokens = action_tokens.squeeze(0)  # (B, E)
        inference_cache["action_tokens"].append(action_tokens[0])
        actions = policy._de_discretize_actions(actions)
        action_bounds = [meta_info["action_bounds"]]
        action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
        action_bounds_high = [
            action_bound["high"] for action_bound in action_bounds
        ]
        action_bounds_low = np.asarray(action_bounds_low)
        action_bounds_high = np.asarray(action_bounds_high)
        action_bounds_low = torch.tensor(
            action_bounds_low, dtype=torch.float32, device=device
        )
        action_bounds_high = torch.tensor(
            action_bounds_high, dtype=torch.float32, device=device
        )
        actions["pose0_position"] = (
            actions["pose0_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
        )
        actions["pose1_position"] = (
            actions["pose1_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
        )
        actions["pose0_position"] = torch.clamp(
            actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
        )
        actions["pose1_position"] = torch.clamp(
            actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
        actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
        actions["pose0_rotation"] = torch.clamp(
            actions["pose0_rotation"], min=-1, max=1
        )
        actions["pose1_rotation"] = torch.clamp(
            actions["pose1_rotation"], min=-1, max=1
        )
        actions = {k: v.cpu().numpy() for k, v in actions.items()}
        actions = any_slice(actions, np.s_[0, 0])
        obs, _, done, info = env.step(actions)
        elapsed_steps += 1
        print(elapsed_steps, done)
        if done:
            break

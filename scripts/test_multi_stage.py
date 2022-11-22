import random
import numpy as np
from vima_bench import make
from vima_bench.tasks.components.encyclopedia import ObjPedia
from vima.utils import *
from vima_bench import *
from example import *
from common import *
from cot_query import load_prompt, query
from logger import Logger
from generate_prompt import prompt_generate
from utils import parse_instruct, task_names


def rollout(policy, task_type, seed, device, prefix='', num_prompts=100, cots=3, num_examples=[2], query_temperature=0.8, render=False):
    logger = Logger(task_type)
    # choose to load prompts or generate new prompts
    # general_instructs = load_prompt(folder=f'prompts/{task_type}_prompts.npy')[:1]  # load prompts
    general_instructs = prompt_generate(type=task_type, num_prompts=num_prompts, num_examples=num_examples)  # generate prompts

    for i, general_instruction in enumerate(general_instructs):
        print(f"Progess: {i}/{num_prompts} instructions\n ------------------------\n")
        # if i % 10 ==0:
        #     render = True
        # else:
        #     render = False
        
        try: 
            for _ in range(cots):
                CoT_prompt = query(general_instruction, temperature=query_temperature)

                task_name = task_names[task_type]
                try:
                    task_kwargs, CoT_prompt, valid = parse_instruct(CoT_prompt, task_type)  # each prompt is a CoT (list of prompts) for one general instruction
                    print(f'CoT: {CoT_prompt}')
                except:
                    valid = False
                    print(f'CoT: {CoT_prompt} cannot be parsed.')
                
                # CoT_prompt = ['Rotate the red L-shaped block 30 degrees.','Rotate the red L-shaped block 10 degrees.','Rotate the red L-shaped block 60 degrees.']
                step_successes = []

                if valid:
                    # make env
                    env = TimeLimitWrapper(
                        ResetFaultToleranceWrapper(
                            make(
                                task_name=task_name,
                                modalities=["segm", "rgb"],
                                task_kwargs=task_kwargs,
                                seed=seed,
                                render_prompt=render,
                                display_debug_window=render,
                                hide_arm_rgb=False,
                                )
                        ),
                        bonus_steps=2,
                    )
                    
                    # run the env for multi-stage manipulation
                    for prompt_step, prompt in enumerate(CoT_prompt): # loop over instruction in CoT
                        env.global_seed = seed
                        # prompt = None
                        keep_scene=False
                        if prompt_step > 0:
                            keep_scene=True
                        obs = env.reset(prompt, keep_scene=keep_scene, task_type=task_type)

                        if render: env.render()

                        meta_info = env.meta_info
                        prompt = env.prompt
                        prompt_assets = env.prompt_assets
                        elapsed_steps = 0
                        inference_cache = {}
                        print('next round: ', prompt)
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
                            success = info['success']
                            print(f'step: {elapsed_steps}, success: {success}')
                            if done:
                                break
                        step_successes.append(success)
                    overall_success = all(step_successes)  # sucess for all prompts is final success
                    logger.track_data(general_instruction, CoT_prompt, overall_success, step_successes)

                    env.close()
                else:
                    print(f'Invalid instruction: {general_instruction}, CoT: {CoT_prompt}!')
                    logger.track_data(general_instruction, CoT_prompt, None, step_successes)
        except:
            print(f'Fail to test with instruction: {general_instruction}, CoT: {CoT_prompt}!')
            env.close()
            continue

        if (i+1) % 20 == 0:
            logger.save(f'data/{prefix}')

if __name__ == "__main__":
    task_type = ['stack', 'rotate', 'put'][1]
    model_size = ['4M', '200M'][-1]
    model_ckpt = f'../models/{model_size}.ckpt'
    device = 'cpu'
    num_prompts=100 # how many instructions to test
    num_examples=[5]
    cots=5 # how many CoTs for each instruction to test
    seed = 42
    policy = create_policy_from_ckpt(model_ckpt, device)
    prefix = f'{model_size}-model_{num_examples}-examples_{num_prompts}-prompts_{cots}-CoTs_verified_'
    rollout(policy, task_type, seed, device, prefix, num_prompts, cots, num_examples)


    # experiments with different number of examples
    # N_examples = [1,3,5]
    # tasks = ['stack', 'rotate', 'put']
    # N_examples = [3,5]
    # tasks = ['rotate']
    # for task_type in tasks:
    #     for N in N_examples:
    #         num_examples = [N]
    #         prefix = f'{model_size}-model_{num_examples}-examples_{num_prompts}-prompts_{cots}-CoTs_'
    #         rollout(policy, task_type, seed, device, prefix, num_prompts, cots, num_examples)

    # experiments with different query temperatures
    # query_temperatures = [0.6, 1.]
    # for query_temperature in query_temperatures:
    #     prefix = f'{model_size}-model_{query_temperature}-temperature_{num_examples}-examples_{num_prompts}-prompts_{cots}-CoTs_'
    #     rollout(policy, task_type, seed, device, prefix, num_prompts, cots, num_examples, query_temperature)

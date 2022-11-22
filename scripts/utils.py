import numpy as np
import random
from vima.utils import *
from vima_bench import *
from example import *
from common import *

task_names = {
    'stack':'visual_manipulation', 
    'put':'visual_manipulation', 
    'rotate': 'rotate'
    }

unuseful_words = ['the', 'The']

def parse_instruct(instruct, type):
    valid = True

    if type == 'rotate':
        # processed_instruct = instruct[4:]  #remove ' A: '
        processed_instruct = instruct[instruct.index('Rotate'):] # start from 'Rotate', for case: '\nRotate ***'
        instruct_list = processed_instruct.split('. ') 
        instruct_list = [instruct for instruct in instruct_list if len(instruct) >2] # filter out empty
        first_instruct = processed_instruct.split('.')[0] # take first of cot
        # Rotate blue triangle by 71 degrees.
        words = first_instruct.split(' ')
        words = [i for i in words if i not in unuseful_words]  # filter

        anchor_pos1 = words.index('Rotate')
        anchor_pos2 = words.index('by')
        color = words[anchor_pos1+1]

        if anchor_pos2 - anchor_pos1 == 4:  # color for 1, two-word obj for 2
            obj = ' '.join(words[anchor_pos1+2:anchor_pos1+4])
        else: # color for 1, one-word obj for 1
            obj = words[anchor_pos1+2]

        # valid check
        if not color in COLORS:
            print(color)
            valid = False
        
        if not obj in OBJECTS:
            print(obj)
            valid = False

        angles = []
        for ins in instruct_list:
            # e.g. Rotate blue triangle by 71 degrees.
            words = ins.split(' ')
            words = [i for i in words if i not in unuseful_words]  # filter
            anchor_pos2 = words.index('by')
            angle = words[anchor_pos2+1]
            angles.append(int(angle))
        # print(words, color, obj, angle)

        # COLORS.remove(color)
        valid_colors = [i for i in COLORS if i not in [color]]
        distractor_color = random.choice(valid_colors)

        task_kwargs = {
        'dragged_obj_express_types': "name",
        'possible_dragged_obj': obj,
        'possible_dragged_obj_texture': [color],
        'possible_distractor_obj_texture': [distractor_color], 
        'specified_angles_of_rotation': angles,
        }

    elif type == 'stack':
        # processed_instruct = instruct[4:]  #remove ' A: '
        processed_instruct = instruct[instruct.index('Put'):] # start from 'Put', for case: '\nPut ***'
        instruct_list = processed_instruct.split('. ')
        instruct_list = [instruct for instruct in instruct_list if len(instruct) >2] # filter out empty 
        drag_obj_list = []
        base_obj_list = []
        drag_color_list = []
        base_color_list = []
        for ins in instruct_list:
            # e.g. Put purple letter E on purple letter M.
            words = ins.split(' ')
            words = [i for i in words if i not in unuseful_words]  # filter

            drag_color = words[words.index('Put')+1]

            if words.index('on') - words.index('Put') == 4:  # color for 1, two-word obj for 2
                drag_obj = ' '.join(words[words.index('Put')+2:words.index('Put')+4])
            else: # color for 1, one-word obj for 1
                drag_obj = words[words.index('Put')+2]
            base_color = words[words.index('on')+1]
            if len(words) - (words.index('on')+2) == 2: # two-word obj
                base_obj = ' '.join(words[words.index('on')+2:words.index('on')+4])
            else: # one-word obj
                base_obj = words[words.index('on')+2]

            drag_obj = drag_obj.replace('.', '')
            base_obj = base_obj.replace('.', '')
            # valid check
            if (not drag_color in COLORS) or (not base_color in COLORS):
                print(drag_color, base_color)
                valid = False
            if (not drag_obj in OBJECTS) or (not base_obj in OBJECTS):
                print(drag_obj, base_obj)
                valid = False
            drag_obj_list.append(drag_obj)
            base_obj_list.append(base_obj)
            drag_color_list.append(drag_color)
            base_color_list.append(base_color)

        valid_colors = [i for i in COLORS if i not in drag_color_list+base_color_list]
        distractor_color = random.choice(valid_colors)

        task_kwargs = {
        'dragged_obj_express_types': "name",
        'base_obj_express_types': 'name',
        'specified_dragged_obj': drag_obj_list,
        'specified_base_obj': base_obj_list[:1],
        'specified_dragged_obj_texture': drag_color_list,
        'specified_base_obj_texture': base_color_list[:1],
        'possible_distractor_obj_texture': distractor_color,
        'num_dragged_obj': len(drag_obj_list),
        }

    elif type == 'put':
        # processed_instruct = instruct[4:]  #remove ' A: '
        processed_instruct = instruct[instruct.index('Put'):] # start from 'Put', for case: '\nPut ***'
        instruct_list = processed_instruct.split('. ') 
        instruct_list = [instruct for instruct in instruct_list if len(instruct) >2] # filter out empty
        drag_obj_list = []
        base_obj_list = []
        drag_color_list = []
        base_color_list = []
        try:
            for i, ins in enumerate(instruct_list):
                # e.g. Put yellow ring into purple container
                words = ins.split(' ')
                words = [i for i in words if i not in unuseful_words]  # filter

                drag_color = words[words.index('Put')+1]
                if words.index('into') - words.index('Put') == 4:  # color for 1, two-word obj for 2
                    drag_obj = ' '.join(words[words.index('Put')+2:words.index('Put')+4])
                else: # color for 1, one-word obj for 1
                    drag_obj = words[words.index('Put')+2] 

                drag_obj = drag_obj.replace('.', '')
                drag_obj_list.append(drag_obj)
                drag_color_list.append(drag_color)
                # valid check
                if (not drag_color in COLORS) or (not drag_obj in OBJECTS):
                    print(drag_color, drag_obj)
                    valid = False
        except:
            print(f'Fail: {instruct_list}')
            valid = False

        # one instruction for base is sufficient (only one base) 
        base_color = words[words.index('into')+1]
        if len(words) - (words.index('into')+2) == 2: # two-word obj
            base_obj = ' '.join(words[words.index('into')+2:words.index('into')+4])
        else: # one-word obj
            base_obj = words[words.index('into')+2] 

        base_obj = base_obj.replace('.', '')
        if (not base_color in COLORS) or (not base_obj in BASE_OBJECTS):
            print(base_color, base_obj)
            valid = False
        base_obj_list.append(base_obj)
        base_color_list.append(base_color)

        valid_colors = [i for i in COLORS if i not in drag_color_list+base_color_list]
        distractor_color = random.choice(valid_colors)
        # import  pdb; pdb.set_trace()

        task_kwargs = {
        'dragged_obj_express_types': "name",
        'base_obj_express_types': 'name',
        'specified_dragged_obj': drag_obj_list,  # using specified, obj-color is one-to-one mapped
        'specified_dragged_obj_texture': drag_color_list,
        'specified_base_obj': base_obj_list,
        'specified_base_obj_texture': base_color_list,
        'possible_distractor_obj_texture': distractor_color,
        'num_dragged_obj': len(drag_obj_list),
        }

    return task_kwargs, instruct_list, valid


def rollout_simulation_with_cot(task_type, CoT_prompt, seed=42, device = 'cpu'):
    # load policy
    model_size = ['4M', '200M'][-1]
    model_ckpt = f'../models/{model_size}.ckpt'
    policy = create_policy_from_ckpt(model_ckpt, device)

    step_successes = []
    task_kwargs, CoT_prompt, valid = parse_instruct(CoT_prompt, task_type)
    task_name = task_names[task_type]
    if valid:
        render = False
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
        
        for prompt_step, prompt in enumerate(CoT_prompt): # loop over instruction in CoT
            env.global_seed = seed
            # prompt = None
            keep_scene=False
            if prompt_step > 0:
                keep_scene=True
            obs = env.reset(prompt, keep_scene=keep_scene, task_type=task_type)

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
        env.close()
    else:
        overall_success = False

    return overall_success
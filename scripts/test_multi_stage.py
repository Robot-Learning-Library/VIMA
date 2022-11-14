import random
from vima_bench import make
from vima_bench.tasks.components.encyclopedia import ObjPedia
from vima.utils import *
from vima_bench import *
from example import *
from common import *
from cot_query import load_prompt, query

task_names = {
    'stack':'visual_manipulation', 
    'put':'visual_manipulation', 
    'rotate': 'rotate'
    }

def parse_instruct(instruct, type):
    if type == 'rotate':
        instruct_list = instruct[4:].split('. ')  #remove ' A: '
        instruct = instruct.split('.')[0] # take first of cot
        # Rotate blue triangle by 71 degrees.
        words = instruct.split(' ')
        color = words[words.index('Rotate')+1]
        if words.index('by') - words.index('Rotate') == 4:  # color for 1, two-word obj for 2
            obj = ' '.join(words[words.index('Rotate')+2:words.index('Rotate')+4])
        else: # color for 1, one-word obj for 1
            obj = words[words.index('Rotate')+2]
        angle = words[words.index('by')+1]
        # print(words, color, obj, angle)

        assert color in COLORS
        COLORS.remove(color)
        distractor_color = random.choice(COLORS)

        task_kwargs = {
        'dragged_obj_express_types': "name",
        'possible_dragged_obj': obj,
        'possible_dragged_obj_texture': [color],
        'possible_distractor_obj_texture': [distractor_color], 
        }

    elif type == 'stack':
        instruct_list = instruct[4:].split('. ')  #remove ' A: '
        drag_obj_list = []
        base_obj_list = []
        drag_color_list = []
        base_color_list = []
        for ins in instruct_list:
            # Put purple letter E on purple letter M.
            words = ins.split(' ')
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
        'specified_base_obj': base_obj_list,
        'specified_dragged_obj_texture': drag_color_list,
        'specified_base_obj_texture': base_color_list,
        'possible_distractor_obj_texture': distractor_color,
        'num_dragged_obj': len(drag_obj_list),
        }

    elif type == 'put':
        instruct_list = instruct[4:].split('. ')  #remove ' A: '
        drag_obj_list = []
        base_obj_list = []
        drag_color_list = []
        base_color_list = []
        try:
            for i, ins in enumerate(instruct_list):
                # Put yellow ring into purple container
                words = ins.split(' ')
                drag_color = words[words.index('Put')+1]
                if words.index('into') - words.index('Put') == 4:  # color for 1, two-word obj for 2
                    drag_obj = ' '.join(words[words.index('Put')+2:words.index('Put')+4])
                else: # color for 1, one-word obj for 1
                    drag_obj = words[words.index('Put')+2] 

                drag_obj = drag_obj.replace('.', '')
                drag_obj_list.append(drag_obj)
                drag_color_list.append(drag_color)
        except:
            print(f'Fail: {instruct_list}')

        # one instruction for base is sufficient (only one base) 
        base_color = words[words.index('into')+1]
        if len(words) - (words.index('into')+2) == 2: # two-word obj
            base_obj = ' '.join(words[words.index('into')+2:words.index('into')+4])
        else: # one-word obj
            base_obj = words[words.index('into')+2] 

        base_obj = base_obj.replace('.', '')
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

    return task_kwargs, instruct_list

# task_kwargs = {
#     'rotate': {
#     'dragged_obj_express_types': "name",
#     'possible_dragged_obj': 'L-shaped block',
#     'possible_dragged_obj_texture': ['red'],  # at least two (two objects), color or TextureEntry
#     'possible_distractor_obj_texture': ['green'], 
#      },

#     'visual_manipulation': {
#     'dragged_obj_express_types': "name",
#     'base_obj_express_types': 'name',
#     'possible_dragged_obj': 'block',
#     # 'possible_base_obj': 'block',
#     'possible_base_obj': 'container',
#     'possible_dragged_obj_texture': ['red and purple polka dot'], # at least one of dragged and base needs to be > 1, to give distractor obj a different color
#     'possible_base_obj_texture': ['green swirl'],
#     'possible_distractor_obj_texture': ['blue'],
#     'num_dragged_obj': 3,
#     }
#     }

# test_prompts = {
#     'rotate': ['Rotate the red L-shaped block 30 degrees.','Rotate the red L-shaped block 10 degrees.','Rotate the red L-shaped block 60 degrees.'],
#     'visual_manipulation': "Put the red and purple polka dot block in the green swirl container."
# }

def rollout(policy, task_type, seed, device):
    general_instructs = load_prompt(folder=f'prompts/{task_type}_prompts.npy')[:1]

    for i, general_instruction in enumerate(general_instructs):
        print(f'{i}-th general instruction: {general_instruction}')
        CoT_prompt = query(general_instruction)

        task_name = task_names[task_type]
        task_kwargs, CoT_prompt = parse_instruct(CoT_prompt, task_type)  # each prompt is a CoT (list of prompts) for one general instruction
        # CoT_prompt = ['Rotate the red L-shaped block 30 degrees.','Rotate the red L-shaped block 10 degrees.','Rotate the red L-shaped block 60 degrees.']
        print(f'CoT prompts: {CoT_prompt}')

        # make env
        env = TimeLimitWrapper(
            ResetFaultToleranceWrapper(
                make(
                    task_name=task_name,
                    modalities=["segm", "rgb"],
                    task_kwargs=task_kwargs,
                    seed=seed,
                    render_prompt=True,
                    display_debug_window=True,
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
            obs = env.reset(prompt, keep_scene=keep_scene)
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
                success = info['success']
                print(f'step: {elapsed_steps}, success: {success}')
                if done:
                    break

        env.close()

if __name__ == "__main__":
    task_type = ['stack', 'rotate', 'put'][0]
    model_ckpt = '../models/200M.ckpt'
    device = 'cpu'

    seed = 42
    policy = create_policy_from_ckpt(model_ckpt, device)
    rollout(policy, task_type, seed, device)

import numpy as np
from common import *
from example import *
from utils import rollout_simulation_with_cot

def sample_discrete(arr, size=1):
    selected = np.random.choice(arr, size)
    return selected

def sample_continuous(arr, size=1):
    selected = np.random.randint(*arr, size)
    return selected

def prompt_construct(type='stack', confs={}, ):
    if type == 'stack':
        prompt = ''
        for j in range(confs['num_examples']):
            num_obj = confs['num_obj'][j]
            ques = "Q: Stack objects in this order: "
            col = sample_discrete(COLORS, size=num_obj)
            obj = sample_discrete(OBJECTS, size=num_obj)
            for i in range(num_obj):
                if i < num_obj-1:
                    ques += col[i] + ' ' + obj[i] + ', '
                else:
                    ques += col[i] + ' ' + obj[i] + '.'

            ans = " A: "
            instrs = ''
            for i in range(num_obj-1):
                instr = STACK_TEMPLATE.replace('DRAG_COLOR', col[i+1])
                instr = instr.replace('BASE_COLOR', col[i])
                instr = instr.replace('DRAG_OBJECT', obj[i+1])
                instr = instr.replace('BASE_OBJECT', obj[i])
                instrs += instr
            prompt += ques + ans + instrs

        # test Q
        num_obj = confs['num_obj'][confs['num_examples']]
        ques = "Q: Stack objects in this order: "
        col = sample_discrete(COLORS, size=num_obj)
        obj = sample_discrete(OBJECTS, size=num_obj)
        for i in range(num_obj):
            if i < num_obj-1:
                ques += col[i] + ' ' + obj[i] + ', '
            else:
                ques += col[i] + ' ' + obj[i] + '.' 
        prompt += ques

    elif type == 'rotate':
        prompt = ''
        verify_through_simulation = False # if True, verify the correctness of the prompt through simulation, only successful prompt will appear in examples
        def get_angles(rot_angle, n):
            rand=np.random.uniform(0,1, n)
            norm_rand=rand/np.sum(rand)*rot_angle
            angles=[int(angle) if i < n else angle for i, angle in enumerate(norm_rand)]
            angles[-1] = rot_angle - sum(angles[:-1]) # make sure the last angle is the remaining angle
            return angles
        for j in range(confs['num_examples']):
            col = sample_discrete(COLORS)[0]
            obj = sample_discrete(OBJECTS)[0]
            rot_angle = sample_continuous(confs['rotate_range'])[0]
            n = np.random.choice(confs['rotate_steps'])
            ques = f"Q: Rotate {col} {obj} by {rot_angle} degrees in {n} steps: "
            
            ans = " A: "
            if verify_through_simulation:
                # load policy
                model_size = ['4M', '200M'][-1]
                model_ckpt = f'../models/{model_size}.ckpt'
                policy = create_policy_from_ckpt(model_ckpt, device='cpu')

                valid = False
                max_iter = 10  # maximum number of iterations to find a valid CoT
                itr = 0
                while not valid and itr < max_iter:
                    itr += 1
                    instrs = ''
                    angles = get_angles(rot_angle, n)  # get n angles, each for one instructionl in CoT
                    for i in range(n):
                        instr = ROTATE_TEMPLATE.replace('DRAG_COLOR', col)
                        instr = instr.replace('DRAG_OBJECT', obj)
                        instr = instr.replace('ANGLE', str(angles[i]))
                        instrs += instr
                    # verify the task is solvable or not
                    valid = rollout_simulation_with_cot('rotate', instrs, policy=policy)
                    # print(f'Verify: {instrs}, Result: {valid}')
                    print(f"Verify for {j}th example, {itr}th iteration: {valid}")
            else:
                instrs = ''
                angles = get_angles(rot_angle, n)
                for i in range(n):
                    instr = ROTATE_TEMPLATE.replace('DRAG_COLOR', col)
                    instr = instr.replace('DRAG_OBJECT', obj)
                    instr = instr.replace('ANGLE', str(angles[i]))
                    instrs += instr

            prompt += ques + ans + instrs

        # test Q
        col = sample_discrete(COLORS)[0]
        obj = sample_discrete(OBJECTS)[0]
        rot_angle = sample_continuous(confs['rotate_range'])[0]
        n = np.random.choice(confs['rotate_steps'])
        ques = f"Q: Rotate {col} {obj} by {rot_angle} degrees in {n} steps: "
        prompt += ques

    elif type == 'put':
        prompt = ''
        for j in range(confs['num_examples']):
            num_obj = confs['num_obj'][j]
            ques = "Q: Put objects X inside the Y. "
            col = sample_discrete(COLORS, size=num_obj)
            obj = sample_discrete(OBJECTS, size=num_obj)
            base_col = sample_discrete(COLORS)[0]
            base_obj = sample_discrete(BASE_OBJECTS)[0]

            placeholder_X = ''
            for i in range(num_obj):
                if i < num_obj-1:
                    placeholder_X += col[i] + ' ' + obj[i] + ', '
                else:
                    placeholder_X += col[i] + ' ' + obj[i]
            placeholder_Y = f'{base_col} {base_obj}'
            ques = ques.replace('X', placeholder_X)
            ques = ques.replace('Y', placeholder_Y)

            ans = " A: "
            instrs = ''
            
            for i in np.random.permutation(num_obj):  # random put order
                instr = PUT_TEMPLATE.replace('DRAG_COLOR', col[i])
                instr = instr.replace('BASE_COLOR', base_col)
                instr = instr.replace('DRAG_OBJECT', obj[i])
                instr = instr.replace('BASE_OBJECT', base_obj)
                instrs += instr
            prompt += ques + ans + instrs

        # test Q
        num_obj = confs['num_obj'][confs['num_examples']]
        ques = "Q: Put objects X inside the Y. "
        col = sample_discrete(COLORS, size=num_obj)
        obj = sample_discrete(OBJECTS, size=num_obj)
        base_col = sample_discrete(COLORS)[0]
        base_obj = sample_discrete(BASE_OBJECTS)[0]

        placeholder_X = ''
        for i in range(num_obj):
            if i < num_obj-1:
                placeholder_X += col[i] + ' ' + obj[i] + ', '
            else:
                placeholder_X += col[i] + ' ' + obj[i]
        placeholder_Y = f'{base_col} {base_obj}'
        ques = ques.replace('X', placeholder_X)
        ques = ques.replace('Y', placeholder_Y)
        
        prompt += ques        

    print(prompt)
    return prompt

def prompt_generate(type ='stack', num_prompts = 5, num_examples = [1,2,3], n_steps=None):

    if n_steps is None:
        n_steps = np.arange(TASK_CONFS[type]['min_steps'], TASK_CONFS[type]['max_steps']+1)

    if type == 'rotate':
        num_objs = [1]

    prompts = []

    for i in range(num_prompts):
        n_example = np.random.choice(num_examples)
        if type == 'stack': # n steps require at least n+1 objects
            p = prompt_construct(type='stack', confs={'num_obj': np.random.choice(n_steps+1, size=n_example+1), 'num_examples': n_example})
        elif type == 'rotate':
            p = prompt_construct(type='rotate', confs={'rotate_range':[90, 180], 'rotate_steps': n_steps, 'num_examples': n_example})
        elif type == 'put':
            p = prompt_construct(type='put', confs={'num_obj': np.random.choice(n_steps, size=n_example+1), 'num_examples': n_example})

        prompts.append(p)


    return prompts

if __name__ == "__main__":

    type = ['stack', 'rotate', 'put'][1]
    prompts = prompt_generate(type)
    np.save(f'prompts/{type}_prompts', prompts)

    prompts = np.load(f'prompts/{type}_prompts.npy')
    print(prompts)
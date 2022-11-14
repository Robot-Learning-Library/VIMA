import numpy as np

STACK_TEMPLATE = "Put DRAG_COLOR DRAG_OBJECT on BASE_COLOR BASE_OBJECT. "
ROTATE_TEMPLATE = "Rotate DRAG_COLOR DRAG_OBJECT by ANGLE degrees. "
PUT_TEMPLATE = "Put DRAG_COLOR DRAG_OBJECT into BASE_COLOR BASE_OBJECT. "

# https://github.com/vimalabs/VIMABench/blob/main/vima_bench/tasks/components/encyclopedia/textures.py
COLORS = ['red', 'blue', 'green', 'orange', 'yellow', 'purple', 'pink', 'cyan', 'olive']
OBJECTS = ['block', 'L-shaped block','letter A','letter E', 'letter G', 'letter M', 'letter R','letter T','letter V', 'cross', 'diamond', 'triangle', 'flower', 'heart', 'hexagon', 'pentagon', 'ring', 'round', 'star']
# BASE_OBJECTS = ['pan', 'bowl', 'frame', 'square', 'container', 'pallet']
BASE_OBJECTS = ['square', 'container']

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
        for j in range(confs['num_examples']):
            col = sample_discrete(COLORS)[0]
            obj = sample_discrete(OBJECTS)[0]
            rot_angle = sample_continuous(confs['rotate_range'])[0]
            n = np.random.choice(confs['rotate_steps'])
            ques = f"Q: Rotate {col} {obj} by {rot_angle} degrees in {n} steps: "
            
            ans = " A: "
            instrs = ''
            rotated_angles = 0
            for i in range(n-1):
                instr = ROTATE_TEMPLATE.replace('DRAG_COLOR', col)
                instr = instr.replace('DRAG_OBJECT', obj)
                angle = sample_continuous(confs['step_range'])[0]
                rotated_angles += angle
                instr = instr.replace('ANGLE', str(angle))
                instrs += instr

            instr = ROTATE_TEMPLATE.replace('DRAG_COLOR', col)
            instr = instr.replace('DRAG_OBJECT', obj)
            angle = rot_angle - rotated_angles
            instr = instr.replace('ANGLE', str(angle))
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
            for i in range(num_obj):
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

def prompt_generate(type ='stack'):
    num_examples = [1,2,3]
    num_prompts = 5

    if type == 'stack':
        num_objs = [2,3,4,5]
    elif type == 'rotate':
        num_objs = [1]
        n_steps = [2,3,4]
        step_range = [0, 45]
    elif type == 'put':
        num_objs = [2,3]

    prompts = []
    for i in range(num_prompts):
        n_example = np.random.choice(num_examples)
        if type == 'stack':
            p = prompt_construct(type='stack', confs={'num_obj': np.random.choice(num_objs, size=n_example+1), 'num_examples': n_example})
        elif type == 'rotate':
            p = prompt_construct(type='rotate', confs={'rotate_range':[90, 180], 'rotate_steps': n_steps, 'step_range': step_range, 'num_examples': n_example})
        elif type == 'put':
            p = prompt_construct(type='put', confs={'num_obj': np.random.choice(num_objs, size=n_example+1), 'num_examples': n_example})

        prompts.append(p)

    return prompts

type = ['stack', 'rotate', 'put'][0]
prompts = prompt_generate(type)
np.save(f'prompts/{type}_prompts', prompts)

prompts = np.load(f'prompts/{type}_prompts.npy')
print(prompts)
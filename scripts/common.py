
STACK_TEMPLATE = "Put DRAG_COLOR DRAG_OBJECT on BASE_COLOR BASE_OBJECT. "
ROTATE_TEMPLATE = "Rotate DRAG_COLOR DRAG_OBJECT by ANGLE degrees. "
PUT_TEMPLATE = "Put DRAG_COLOR DRAG_OBJECT into BASE_COLOR BASE_OBJECT. "

# https://github.com/vimalabs/VIMABench/blob/main/vima_bench/tasks/components/encyclopedia/textures.py
COLORS = ['red', 'blue', 'green', 'orange', 'yellow', 'purple', 'pink', 'cyan', 'olive']
OBJECTS = ['block', 'L-shaped block','letter A','letter E', 'letter G', 'letter M', 'letter R','letter T','letter V', 'cross', 'diamond', 'triangle', 'flower', 'heart', 'hexagon', 'pentagon', 'ring', 'round', 'star']
# BASE_OBJECTS = ['pan', 'bowl', 'frame', 'square', 'container', 'pallet']
BASE_OBJECTS = ['square', 'container']

TASK_CONFS = {
    'stack': {
        'min_steps': 2,
        'max_steps': 5,
    },
    'rotate': {
        'min_steps': 2,
        'max_steps': 4,
    },
    'put': {
        'min_steps': 2,
        'max_steps': 3,
    },

 }
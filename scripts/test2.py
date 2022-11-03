from vima_bench import make
import time

task_kwargs = {
    'dragged_obj_express_types': "name",
    'base_obj_express_types': 'name',
    'possible_base_obj': 'block',
    'possible_dragged_obj': 'block',
    }
env = make(task_name="visual_manipulation", 
        task_kwargs=task_kwargs,
        render_prompt=True,
        display_debug_window=True,
        hide_arm_rgb=False,
        )

obs = env.reset()
env.render()
# time.sleep(10)
prompt, prompt_assets = env.prompt, env.prompt_assets
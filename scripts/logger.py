import pandas as pd
from common import TASK_CONFS

class Logger():
    def __init__(self, type=''):
        self.type = type
        self.data = {
            'question': [],
            'general_instruct': [], 
            'CoT_prompt': [],
            'steps': [],
            'overall_success': [],
            }
        self.max_steps = TASK_CONFS[self.type]['max_steps']

        for i in range(self.max_steps):  # max steps of task
            self.data[f'{i}-step_success'] = []

        # add task specific 

    def track_data(self, general_instruct, CoT_prompt, success, step_successes):
        question = general_instruct.split('Q: ')[-1]
        self.data['question'].append(question)
        self.data['general_instruct'].append(general_instruct)
        self.data['CoT_prompt'].append(CoT_prompt)
        self.data['steps'].append(len(CoT_prompt))
        self.data['overall_success'].append(success)

        step_successes = step_successes + [None] * (self.max_steps - len(step_successes))  # fill empty

        for i in range(self.max_steps):  # max steps of task
            self.data[f'{i}-step_success'].append(step_successes[i])

        # add task specific 

    def save(self, folder=''):
        df = pd.DataFrame(self.data)
        df.to_csv(folder+f'{self.type}_results.csv', index=False, header=True)

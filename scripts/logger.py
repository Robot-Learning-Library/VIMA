import pandas as pd


class Logger():
    def __init__(self, ):
        self.data = {
            'general_instruct': [], 
            'CoT_prompt': [],
            'success': [],
            }

    def track_data(self, general_instruct, CoT_prompt, success):
        self.data['general_instruct'].append(general_instruct)
        self.data['CoT_prompt'].append(CoT_prompt)
        self.data['success'].append(success)

    def save(self, folder=''):
        df = pd.DataFrame(self.data)
        df.to_csv(folder+'results.csv', index=False, header=True)

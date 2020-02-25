import os

class Experiment():
    
    def __init__(self):
        self.nas_main_dir = os.environ['CG_NAS_MAIN']
        self.oct_main_dir = os.environ['CG_OCT_MAIN']
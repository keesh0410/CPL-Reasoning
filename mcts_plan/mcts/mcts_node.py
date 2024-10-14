import numpy as np


class MCTSNode:

    depth = 0
    is_terminal = False # don't need to expand
    value = -100
    tag = "0"

    prior = 1.0
    c_puct = 1.5


    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.state = {"prompt": "", "text": "", "step_plan": "", "solution": "", "final_answer": ""}   
        self.children = []
        self.__value_sum = 0.0
        self.__visit_count = 0
        

    def has_children(self):
        return len(self.children) > 0

    def is_root(self):
        return self.parent is None

    def q_value(self):
        if self.__visit_count == 0:
            return 0
        return self.__value_sum / self.__visit_count
    
    def visit_count(self):
        return self.__visit_count
    
    def update(self, value):
        if self.value == -100:
            self.value = value
        self.__visit_count += 1
        self.__value_sum += value
    
    def update_recursive(self, value, start_node):
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)
    
    def puct(self):
        q_value = self.q_value() if self.visit_count() > 0 else 0
        u_value = self.c_puct * self.prior * np.sqrt(self.parent.visit_count()) / (1 + self.visit_count())
        return q_value + u_value

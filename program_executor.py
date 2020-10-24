import numpy as np
import pandas as pd

class DSL():
    '''Domain Specific Language consisting of functions and relations'''
    def __init__(self):
        # Value to Attribute Converter
        self.val2attr = {'shape': ['circle', 'rectangle'], 'color': ['red', 'green', 'blue', 'orange', 'gray', 'yellow']}
        self.get_attr = lambda x: [k for k, v in self.val2attr.items() if x in v][0]
        
        # String to Function
        self.str2func = {'filter': self.filter_, 
                         'query': self.query, 
                         'count': self.count, 
                         'relate': self.relate, 
                         'isleft': self.isLeft, 
                         'istop': self.isTop}
        
    def filter_(self, param):
        '''Returns a subset of the scene based on the param'''    
        # Filter Object(s) for scene
        attr = self.get_attr(param)
        filtered_objects = self.scene[self.scene[attr] == param]

        return filtered_objects
    
    def query(self, obj, attr):
        '''Returns the column value of object(s)'''
        i = obj.index[0]
        result = obj[attr][i]

        return result
    
    def relate(self, obj, param):
        '''Returns object which is either closest or furthest from all other objects of the scene'''
        obj_pos = self.query(obj, 'position')
        scene_pos = self.scene['position']

        # Calculate distances
        distances = np.array([np.linalg.norm(np.array(obj_pos) - np.array(pos)) for pos in scene_pos])
        
        sorted_dists = distances.argsort()
            
        if param == 'closest':
            idx = sorted_dists[0] if distances[sorted_dists[0]] != 0 else sorted_dists[1]
        elif param == 'furthest':
            idx = sorted_dists[-1] if distances[sorted_dists[-1]] != 0 else sorted_dists[-2]
        
#         print(sorted_dists, distances, idx)

        # Get the object from the scene of that index
        req_obj = self.scene[self.scene.index == idx]

        return req_obj
    
    def count(self, objects):
        '''Counts the objects'''
        return len(objects)
    
    def isLeft(self, pos):
        '''Checks if a position is on the left half or not'''
        return 'yes' if pos[0] < 112 else 'no'
    
    def isTop(self, pos):
        '''Checks if a position is on the top half or not'''
        return 'yes' if pos[1] < 112 else 'no'

class ProgramExecutor(DSL):
    '''Executes a given program'''
    def __init__(self):
        super().__init__()
        pass
    
    def func_executor(self, func, param, prev_out):
        '''Executes a given function with or without a parameter'''
        # 0-1 arg functions
        if func in ['filter']:
            prev_out = self.filter_(param) if param != None else self.filter_(prev_out)

        # Two arg functions
        elif func in ['query', 'relate']:
            prev_out = self.str2func[func](prev_out, param)

        # One arg functions
        elif func in ['count', 'isleft', 'istop']:
            prev_out = self.str2func[func](prev_out)

        return prev_out
    
    def __call__(self, scene, program):
        '''Executes a program on the scene'''
        self.scene = scene
        
        prev_out = None
        for seq in program:
            args = seq.split()
            # print(args)
#             try:
            if len(args) < 2:
                prev_out = self.func_executor(args[0], None, prev_out)
            else:
                prev_out = self.func_executor(args[0], args[1], prev_out)
            # print(prev_out, '\n')
#             except:
#                 prev_out = 'Failed'

        return prev_out
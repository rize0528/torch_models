import os
from .LinearRegression import LinearRegression

wd = os.path.dirname(os.path.realpath(__file__))
available_models = list(map(lambda x: x.replace('.py', ''),
                            [x for x in os.listdir(wd) if not x.startswith('_')]))


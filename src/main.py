#from src.evaluation.test import test
from training.train import train


session_name = train("batch16")
# test(session_name=session_name, is_visualize=False)
# test(session_name='test', is_visualize=True)

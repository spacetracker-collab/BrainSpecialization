
# Fixed Brain Specialization Project

## Run
pip install torch matplotlib
python train.py



from train import train
from visualize import plot_history
from detector import detect_phase_transition

h = train()
plot_history(h)
print(detect_phase_transition(h)[:10])

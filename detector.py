
def detect_phase_transition(history, ratio=1.5):
    return [i for i,(l,s,c) in enumerate(history) if s > ratio*c]

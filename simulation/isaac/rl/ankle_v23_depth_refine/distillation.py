"""Keep the frozen teacher's recurrent state continuous across PPO-sized chunks."""
from rsl_rl.algorithms import Distillation


def clone_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(clone_state(value) for value in state)
    return state.detach().clone()


class ContinuousTeacherDistillation(Distillation):
    def update(self):
        # The installed base algorithm resets teacher state during the student
        # replay loop without replaying the teacher. Preserve the already
        # collected teacher state, including terminal-episode resets.
        teacher_state = clone_state(self.teacher.get_hidden_state())
        result = super().update()
        self.teacher.reset(hidden_state=teacher_state)
        self.last_hidden_states = (self.last_hidden_states[0], teacher_state)
        return result

from QLearning import QLearning


class MyQLearning(QLearning):

    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        maxQSPrime = max(self.get_action_values(state_next, possible_actions))
        value = self.get_q(state, action) + (alfa * (r + gamma * maxQSPrime - self.get_q(state, action)))
        self.set_q(state, action, value)

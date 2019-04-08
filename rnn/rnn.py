import numpy as np

def simple_rnn():
    # init input and state
    X = [1, 2]
    state = [0.0, 0.0]

    # cell weight & bias
    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    w_cell_input = np.asarray([0.5, 0.6])
    b_cell = np.asarray([0.1, -0.1])

    # output weight & bias
    w_output = np.asarray([[1.0], [2.0]])
    b_output = 0.1

    for i in range(len(X)):
        before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
        state = np.tanh(before_activation)
        final_output = np.dot(state, w_output) + b_output

        print('before activation', before_activation)
        print('state', state)
        print('output', final_output)


if __name__ == '__main__':
    simple_rnn()

import theano
import theano.tensor as T


def skew(inputs, backwards=False):
    """
    :param inputs: shape is (minibatch, channels, height, width)
    :param backwards: skew the other way
    :return: skewed image
    """
    skewed = T.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 2 * inputs.shape[3] - 1), theano.config.floatX)
    height = inputs.shape[2]

    def skew_row(i, j, input_row, skewed):
        width = inputs.shape[3]
        skewed = T.set_subtensor(skewed[:, :, i, j:j + width], input_row)
        return skewed

    if not backwards:
        skewed_indices = T.arange(height)
    else:
        # in reverse
        skewed_indices = T.arange(height)[::-1]
    outputs, _ = theano.scan(fn=skew_row,
                             sequences=[T.arange(height), skewed_indices, inputs.dimshuffle((2, 0, 1, 3))],
                             outputs_info=skewed)
    result = outputs[-1]
    return result


def unskew(inputs, backwards=False):
    width = (inputs.shape[3] + 1) // 2
    height = inputs.shape[2]
    unskewed = T.zeros((inputs.shape[0], inputs.shape[1], height, width))

    def unskew_row(i, j, unskewed, inputs):
        unskewed = T.set_subtensor(unskewed[:, :, i, :], inputs[:, :, i, j:j + width])
        return unskewed

    if not backwards:
        indices_skewed = T.arange(height)
    else:
        # in reverse
        indices_skewed = T.arange(height)[::-1]
    outputs, _ = theano.scan(fn=unskew_row,
                             sequences=[T.arange(height), indices_skewed],
                             outputs_info=unskewed,
                             non_sequences=inputs)
    result = outputs[-1]
    return result


if __name__ == "__main__":
    import numpy as np

    input = T.tensor4("unskewed")
    f = theano.function(inputs=[input], outputs=skew(input, backwards=False))
    f_unskew = theano.function(inputs=[input], outputs=unskew(input, backwards=False))
    in_arr = np.ones((1, 1, 6, 6), dtype=np.float32)
    in_arr[:, :, 3, 4] = 0
    in_arr[:, :, 1, 3] = 0
    skewed_input = f(in_arr)
    print(skewed_input)
    print(f_unskew(skewed_input))

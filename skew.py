import theano
import theano.tensor as T


def skew(inputs):
    """
    :param inputs: shape is (minibatch, channels, height, width)
    :param backwards: skew the other way
    :return: skewed image
    """
    skewed = T.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 2 * inputs.shape[3] - 1), theano.config.floatX)
    height = inputs.shape[2]

    def skew_row(i, input_row, skewed):
        width = inputs.shape[3]
        skewed = T.set_subtensor(skewed[:, :, i, i:i + width], input_row)
        return skewed

    outputs, _ = theano.scan(fn=skew_row,
                             sequences=[T.arange(height), inputs.dimshuffle((2, 0, 1, 3))],
                             outputs_info=skewed)
    result = outputs[-1]
    return result


def unskew(inputs):
    width = (inputs.shape[3] + 1) // 2
    height = inputs.shape[2]
    unskewed = T.zeros((inputs.shape[0], inputs.shape[1], height, width))

    def unskew_row(i, unskewed, inputs):
        unskewed = T.set_subtensor(unskewed[:, :, i, :], inputs[:, :, i, i:i + width])
        return unskewed

    outputs, _ = theano.scan(fn=unskew_row,
                             sequences=[T.arange(height)],
                             outputs_info=unskewed,
                             non_sequences=inputs)
    result = outputs[-1]
    return result


if __name__ == "__main__":
    import numpy as np

    input = T.tensor4("unskewed")
    f = theano.function(inputs=[input], outputs=skew(input))
    f_unskew = theano.function(inputs=[input], outputs=unskew(input))
    in_arr = np.arange(36, dtype=np.float32).reshape((1, 1, 6, 6))
    skewed_input = f(in_arr)
    print(skewed_input)
    print(f_unskew(skewed_input))

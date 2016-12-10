import theano
import theano.tensor as T
import lasagne
from skew import skew, unskew


class DiagLSTMLayer(lasagne.layers.Layer):
    def __init__(self, incoming, K_ss=lasagne.init.GlorotUniform(), backwards=False, **kwargs):
        super(DiagLSTMLayer, self).__init__(incoming, **kwargs)

        self.K_ss = self.add_param(K_ss, (self.input_shape[1], self.input_shape[1] // 4, 2), name="K_ss")
        self.b = self.add_param(lasagne.init.Constant(0.), (1,), name="K_ss_bias", regularizable=False)
        self.backwards = backwards

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1] // 4, input_shape[2], input_shape[3]

    def get_output_for(self, input_to_state, **kwargs):
        # skew the input in the right direction
        if self.backwards:
            input_to_state = input_to_state[:, :, :, ::-1]

        skewed = skew(input_to_state)
        K_ss = self.K_ss
        b = self.b
        batch_size = self.input_shape[0]
        in_chan_dim = self.input_shape[1] // 4
        height = self.input_shape[2]

        def process_column(x, c_prev, h_prev):
            # dim (batch_size x in_chan_dim x height)
            column_in = lasagne.layers.InputLayer(input_var=h_prev, shape=(batch_size, in_chan_dim, height))

            # OK, conv1d with filter_size (2,) puts the value at the second position of the conv.
            # Which is ok for me, as long as I process the columns from top to bottom.
            convolved_states = lasagne.layers.Conv1DLayer(incoming=column_in, num_filters=4 * in_chan_dim,
                                                          filter_size=(2,),
                                                          W=K_ss, b=b,
                                                          pad="full",
                                                          nonlinearity=lasagne.nonlinearities.identity)
            convolved_states = lasagne.layers.get_output(convolved_states)
            # "full" adds one unneeded element at the end for filter_size=2
            convolved_states = convolved_states[:, :, :-1]

            # the input x is already convolved at this point
            lstm_parts = convolved_states + x

            o = T.nnet.sigmoid(lstm_parts[:, 0:in_chan_dim])
            f = T.nnet.sigmoid(lstm_parts[:, in_chan_dim:2 * in_chan_dim])
            i = T.nnet.sigmoid(lstm_parts[:, 2 * in_chan_dim:3 * in_chan_dim])
            g = T.tanh(lstm_parts[:, 3 * in_chan_dim:])

            c = f * c_prev + i * g
            h = o * T.tanh(c)

            return c, h  # dims of both are: (batch_size x in_chan_dim x height)

        column_shape = (skewed.shape[0], skewed.shape[1] // 4, skewed.shape[2])
        outputs, updates = theano.scan(fn=process_column,
                                       sequences=skewed.dimshuffle((3, 0, 1, 2)),
                                       outputs_info=[T.zeros(column_shape, dtype=theano.config.floatX),
                                                     T.zeros(column_shape, dtype=theano.config.floatX)],
                                       allow_gc=True)
        _, hs = outputs
        hs = hs.dimshuffle((1, 2, 3, 0))
        hs = unskew(hs)
        if self.backwards:
            # we need to reverse the columns again
            hs = hs[:, :, :, ::-1]
        return hs


if __name__ == "__main__":
    import numpy as np

    in_tensor = T.tensor3("in")
    in_layer = lasagne.layers.InputLayer(input_var=in_tensor, shape=(1, 1, 5))
    out = lasagne.layers.Conv1DLayer(incoming=in_layer, num_filters=1,
                                     filter_size=(2,),
                                     W=np.ones((1, 1, 2), dtype=np.float32), b=lasagne.init.Constant(0.),
                                     pad="full",
                                     nonlinearity=lasagne.nonlinearities.identity)
    out_tensor = lasagne.layers.get_output(out)
    f = theano.function(inputs=[in_tensor], outputs=out_tensor)

    print(f(np.ones((1, 1, 5), dtype=np.float32)))
    # TODO: test the LSTM layer

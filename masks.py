import numpy as np
import theano
import theano.tensor as T
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


def mask_spatial_context(filter):
    filter_side = filter.shape[-1]
    half = filter_side // 2
    # zero out context connections
    filter = T.set_subtensor(filter[:, :, half + 1:, :], 0)
    filter = T.set_subtensor(filter[:, :, half, half + 1:], 0)
    return filter


def mask_a(filter):
    """
    :param filter: 4d symbolic filter
            (output_channels x input_channels x height x width)
    :return: masked filter
    """

    filter = mask_spatial_context(filter)

    # mask the connections between input channels and output channels
    # order of channels is RGB
    out_chan_part = filter.shape[0] // 3
    in_chan_part = filter.shape[1] // 3
    center = filter.shape[-1] // 2

    R_out = filter.shape[0] - 2*out_chan_part
    RG_out = R_out + out_chan_part
    RGB_out = RG_out + out_chan_part

    R_in = filter.shape[1] - 2*in_chan_part
    RG_in = R_in + in_chan_part

    # R_out does not see any of the input channels
    filter = T.set_subtensor(filter[:R_out, :, center, center], 0)
    # G_out does not see anything after the R input channel
    filter = T.set_subtensor(filter[R_out:RG_out, R_in:, center, center], 0)
    # B_out does not see anything after the R and G input channels
    filter = T.set_subtensor(filter[RG_out:RGB_out, RG_in:, center, center], 0)

    return filter


def mask_b(filter):
    """
        :param filter: 4d symbolic filter
                (output_channels x input_channels x height x width)
        :return: masked filter
        """

    filter = mask_spatial_context(filter)

    # mask the connections between input channels and output channels
    # order of channels is RGB
    out_chan_part = filter.shape[0] // 3
    in_chan_part = filter.shape[1] // 3
    center = filter.shape[-1] // 2

    R_out = filter.shape[0] - 2*out_chan_part
    RG_out = R_out + out_chan_part

    R_in = filter.shape[1] - 2*in_chan_part
    RG_in = R_in + in_chan_part

    # R_out does not see anything after R_in
    filter = T.set_subtensor(filter[:R_out, R_in:, center, center], 0)
    # G_out does not see anything after the G_in
    filter = T.set_subtensor(filter[R_out:RG_out, RG_in:, center, center], 0)
    # B_out does sees it all, no need to zero it out

    return filter


if __name__ == "__main__":
    input = T.tensor4("filter")
    output = mask_a(input)
    f = theano.function(inputs=[input], outputs=output)

    masked = f(np.ones((3, 3, 3, 3), dtype=np.float32))

    log.info("red out")
    log.info(masked[0])

    log.info("green out")
    log.info(masked[1])

    log.info("blue out")
    log.info(masked[2])

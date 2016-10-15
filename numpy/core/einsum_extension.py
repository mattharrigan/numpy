import numpy as np
import re


def einsum_elementwise(subscripts, *operands, **kwargs):
    """
    Wrapper around einsum to allow for elementwise addition and subtraction.  Adds support for "+" and "-" characters,
    which separate current einsum statements and then combine with elementwise addition or subtraction.  The arrays to
    be combined must have equal shape, no broadcasting.  Currently this functionality is only for the string API.

    :param subscripts:
    :param operands:
    :param kwargs:
    :return:
    """

    # currently only the string API is parsed for elementwise addition/subtraction.  If the input is from the other API,
    # just call the original function
    # TODO: make it work for both API's
    if not isinstance(subscripts, str):
        return np.einsum(subscripts, *operands, **kwargs)

    # split the subscripts on each + or - but not ->, keeping the + or - characters
    # the even segments are calls to einsum and the odd segments indicate how to combine them
    segments = re.split(r'([+-](?!>))', subscripts)

    # check for common errors in the subscripts string
    if segments[0] == '+' or segments[0] == '-':
        raise ValueError('subscripts cannot start with a + or -')
    if segments[-1] == '+' or segments[-1] == '-':
        raise ValueError('subscripts cannot end with a + or -')
    for segment in segments:
        # re.split puts empty strings between repeated split patterns
        if not segment:
            raise ValueError('subscripts cannot have repeated + or -')

    # the operands for the first segment are operands[:stop]
    # there is 1 more operand than comma in a segment
    stop = segments[0].count(',') + 1

    # compute the output of the first segment, using the output array if provided
    if 'out' in kwargs:
        result = kwargs['out']
        np.einsum(segments[0], *operands[:stop], **kwargs)
    else:
        result = np.einsum(segments[0], *operands[:stop], **kwargs)

    # loop over remaining segments
    for i in range(1, len(segments), 2):

        # the current segment's operands will start where the last segment's operands stopped
        start = stop

        # increment stop by the number of operands for this segment
        stop += segments[i+1].count(',') + 1

        # compute the result for this segment
        curr_segment_result = np.einsum(segments[i+1], *operands[start: stop], **kwargs)

        # Make sure that the arrays have equal shape.
        # TODO: confirm or modify, allowing for broadcasting seems not in spirit with the rest of einsum and might
        # complicate future lower level implementations and optimizations.  Broadcasting wont work with this
        # implementation, if the shape of result changes during the operation it cant be done in place
        if curr_segment_result.shape != result.shape:
            raise ValueError('Arrays for elementwise addition or subtraction must have equal shapes')

        # update the result with this segment's result
        if segments[i] == '+':
            np.add(result, curr_segment_result, out=result, casting=kwargs.get('casting', 'safe'))
        else:
            np.subtract(result, curr_segment_result, out=result, casting=kwargs.get('casting', 'safe'))

    return result

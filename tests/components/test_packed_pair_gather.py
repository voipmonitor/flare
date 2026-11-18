"""Packed routing derives its scratch size and rejects overlapping or incompatible storage."""

import pytest
import torch

from ignition.components.packed_pair_gather import packed_pair_geometry


def operands(rows):
    return (torch.empty((rows, 224), dtype=torch.bfloat16), torch.empty((rows, 56)),
            torch.empty((rows, 3584), dtype=torch.bfloat16), torch.empty((rows, 896)))


@pytest.mark.parametrize('rows,local,offset,total', [(4, 2688, 2816, 45824), (7, 4704, 4864, 80128),
                                                   (8, 5376, 5376, 91392)])
def test_byte_transport_preserves_field_widths_and_aligns_only_the_receive_buffer(rows, local, offset, total):
    assert packed_pair_geometry(*operands(rows), world=16, channel_bytes=672) == (112, 56, local, offset, total)


def test_output_alias_is_rejected_before_any_gather():
    _, second, output, other_output = operands(7)
    first = output.flatten()[:7 * 224].view(7, 224)
    with pytest.raises(ValueError, match='overlap'):
        packed_pair_geometry(first, second, output, other_output, world=16, channel_bytes=672)


@pytest.mark.parametrize('change', ['dtype', 'rows', 'channel', 'alignment'])
def test_incompatible_transport_geometry_is_rejected(change):
    first, second, out_first, out_second = operands(7)
    channel = 672
    if change == 'dtype':
        second = second.to(torch.bfloat16)
    elif change == 'rows':
        second = second[:6]
    elif change == 'channel':
        channel = 688
    else:
        first = torch.empty((7, 226), dtype=torch.bfloat16)
        out_first = torch.empty((7, 226 * 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        packed_pair_geometry(first, second, out_first, out_second, world=16, channel_bytes=channel)

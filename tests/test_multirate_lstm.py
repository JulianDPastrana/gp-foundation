import torch
import pytest
from chainedgp.rnn import MultiRateLSTM

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BATCH_SIZE = 2
HIDDEN_SIZE = 4
INPUT_SIZES = [1, 2, 3, 4, 6]  # also your input_size_list


# ─── DEVICE FIXTURE ────────────────────────────────────────────────────────────
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def random_seqs():
    return [
        torch.randn(5, BATCH_SIZE, 3),
        torch.randn(10, BATCH_SIZE, 1),
        torch.randn(2, BATCH_SIZE, 2),
    ]


# ─── TESTS ─────────────────────────────────────────────────────────────────────
def test_output_and_state_shapes(device):
    seqs = [s.to(device) for s in random_seqs()]
    model = MultiRateLSTM(input_size_list=INPUT_SIZES, hidden_size=HIDDEN_SIZE).to(
        device
    )

    out, (h_n, c_n) = model(seqs)
    max_len = max(s.size(0) for s in seqs)

    assert out.shape == (max_len, BATCH_SIZE, HIDDEN_SIZE)
    assert h_n.shape == (BATCH_SIZE, HIDDEN_SIZE)
    assert c_n.shape == (BATCH_SIZE, HIDDEN_SIZE)


def test_all_cells_registered():
    model = MultiRateLSTM(input_size_list=INPUT_SIZES, hidden_size=HIDDEN_SIZE)
    registered = set(int(k) for k in model.cells.keys())
    assert registered == set(INPUT_SIZES)


def test_order_invariance(device):
    seqs = random_seqs()
    seqs = [s.to(device) for s in seqs]
    model = MultiRateLSTM(INPUT_SIZES, HIDDEN_SIZE).to(device)

    out1, (h1, c1) = model(seqs)
    # shuffle the list order
    out2, (h2, c2) = model([seqs[1], seqs[2], seqs[0]])

    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.allclose(h1, h2, atol=1e-6)
    assert torch.allclose(c1, c2, atol=1e-6)


def test_single_sequence_behaviour(device):
    # pick one input sequence
    seq = random_seqs()[0].to(device)
    input_size = seq.size(-1)

    # build and run the multi‐rate model
    model = MultiRateLSTM([input_size], HIDDEN_SIZE).to(device)
    multi_out, (mh, mc) = model([seq])

    # grab the exact same cell inside the model
    cell = model.cells[str(input_size)]

    # now manually step that cell over the sequence
    h = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, device=device)
    c = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, device=device)
    expected_outputs = []
    for t in range(seq.size(0)):
        h, c = cell(seq[t], (h, c))
        expected_outputs.append(h)
    expected = torch.stack(expected_outputs, dim=0)

    # compare shapes and numerical equality
    assert multi_out.shape == expected.shape
    assert torch.allclose(multi_out, expected, atol=1e-6)

    # final states should also match
    assert torch.allclose(mh, h, atol=1e-6)
    assert torch.allclose(mc, c, atol=1e-6)

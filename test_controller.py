from scope_utils import vg_setrel

def test_vg_setrel():
    assert vg_setrel(2500, 500, 0) == 2750
    assert vg_setrel(2500, 500, 100) == 2850
    assert vg_setrel(2500, 500, -100) == 2650
    assert vg_setrel(2500, 400, 0) == 2700
    assert vg_setrel(2500, 400, 100) == 2800
    assert vg_setrel(2500, 400, -100) == 2600

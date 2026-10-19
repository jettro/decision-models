import importlib

import pytest

from decision_models import main
from decision_models.samples import SAMPLES, module_name


@pytest.mark.parametrize('sample', SAMPLES)
def test_every_sample_has_a_main(sample):
    module = importlib.import_module(module_name(sample))

    assert callable(module.main)


def test_without_a_sample_lists_them(capsys):
    main([])

    output = capsys.readouterr().out
    for sample in SAMPLES:
        assert sample in output


def test_unknown_sample_is_rejected():
    with pytest.raises(SystemExit):
        main(['does-not-exist'])

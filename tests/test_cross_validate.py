import pytest
import os

from gpseer.cross_validate import (
    main
)

# We'll use the `pytest.mark.datafiles` decorator to
# autoload temporary data files form the example
# directory at the top level of this repo.
@pytest.mark.datafiles(
    'examples/example-full.csv'
)
def test_main(
    logger,
    console_log,
    tmp_path,
    datafiles
):
    infile = datafiles / 'example-full.csv'
    outroot = tmp_path / 'cv'
    n_samples = 5

    main(
        logger,
        str(infile),
        n_samples,
        output_root=str(outroot),
    )

    expected_outputs = ["_cross-validation-scores.csv",
                        "_correlation-plot.pdf"]
    for e in expected_outputs:
        outfile = "{}{}".format(outroot,e)
        assert os.path.isfile(outfile)

    # Assert logging is working
    main_out = "{}_cross-validation-scores.csv".format(outroot)
    console = console_log.getvalue()
    assert f"Reading data from {infile}..." in console
    assert "Sampling the data..." in console
    assert f"Writing scores to {main_out}..." in console
    assert "GPSeer finished!" in console

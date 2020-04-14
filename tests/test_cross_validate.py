import pytest

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
    outfile = tmp_path / 'example-scores.csv'
    n_samples = 5

    main(
        logger,
        str(infile),
        n_samples,
        output_file=str(outfile),
    )

    assert outfile.is_file()

    # Assert logging is working
    console = console_log.getvalue()
    assert f"Reading data from {infile}..." in console
    assert "Sampling the data..." in console
    assert f"Writing scores to {outfile}..." in console
    assert "GPSeer finished!" in console

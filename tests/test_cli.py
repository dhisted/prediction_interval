from src.prediction_interval import cli


def test_cli_template():
    assert cli.cli() is None

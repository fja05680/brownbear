import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import brownbear as bb
from brownbear.investment_options import (
    compute_investment_metrics,
    format_investment_options_from_universe,
    format_investment_options_lines,
    read_symbols_from_input,
    update_investment_options,
)


class TestInvestmentOptions(unittest.TestCase):
    def test_read_symbols_from_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'investment-options-in.csv'
            path.write_text(
                '# comment\n'
                '"Investment Option","Description","Asset Class"\n'
                '"SPY","S&P 500","US Stocks"\n'
                '"QQQ","Nasdaq","US Stocks"\n'
            )
            self.assertEqual(read_symbols_from_input(path), ['QQQ', 'SPY'])

    def test_format_investment_options_lines(self):
        metrics = {
            'annual_returns_3mo': pd.Series({'SPY': 1.0}),
            'annual_returns_6mo': pd.Series({'SPY': 2.0}),
            'annual_returns_1yr': pd.Series({'SPY': 3.0}),
            'annual_returns_1_1yr': pd.Series({'SPY': 4.0}),
            'annual_returns_3yr': pd.Series({'SPY': 5.0}),
            'annual_returns_5yr': pd.Series({'SPY': 6.0}),
            'vola': pd.Series({'SPY': 0.1}),
            'ds_vola': pd.Series({'SPY': 0.08}),
            'std_dev_1yr': pd.Series({'SPY': 0.11}),
            'std_dev_3yr': pd.Series({'SPY': 0.12}),
            'std_dev_5yr': pd.Series({'SPY': 0.13}),
        }
        lines = [
            '# header comment',
            '"Investment Option","Description","Asset Class","3 mo"',
            '"SPY","S&P 500","US Stocks",""',
        ]
        out = format_investment_options_lines(lines, metrics)
        self.assertEqual(out[0], '# header comment')
        self.assertIn('"SPY","S&P 500","US Stocks","1.00","2.00"', out[2])

    def test_format_investment_options_from_universe(self):
        metrics = {
            'annual_returns_3mo': pd.Series({'SPY': 1.0}),
            'annual_returns_6mo': pd.Series({'SPY': 2.0}),
            'annual_returns_1yr': pd.Series({'SPY': 3.0}),
            'annual_returns_1_1yr': pd.Series({'SPY': 4.0}),
            'annual_returns_3yr': pd.Series({'SPY': 5.0}),
            'annual_returns_5yr': pd.Series({'SPY': 6.0}),
            'vola': pd.Series({'SPY': 0.1}),
            'ds_vola': pd.Series({'SPY': 0.08}),
            'std_dev_1yr': pd.Series({'SPY': 0.11}),
            'std_dev_3yr': pd.Series({'SPY': 0.12}),
            'std_dev_5yr': pd.Series({'SPY': 0.13}),
        }
        universe = pd.DataFrame(
            {'Description': ['S&P 500'], 'Asset Class': ['US Stocks']},
            index=['SPY'],
        )
        out = format_investment_options_from_universe(
            ['# Description: S&P 500 Galaxy', '"Investment Option","Description","Asset Class"'],
            universe,
            metrics,
        )
        self.assertEqual(len(out), 3)
        self.assertIn('"SPY","S&P 500","US Stocks"', out[2])

    def test_format_falls_back_when_longer_returns_missing(self):
        metrics = {
            'annual_returns_3mo': pd.Series({'SPY': 1.0}),
            'annual_returns_6mo': pd.Series({'SPY': 2.0}),
            'annual_returns_1yr': pd.Series({'SPY': 3.0}),
            'annual_returns_1_1yr': pd.Series({'SPY': 4.0}),
            'annual_returns_3yr': pd.Series({'SPY': np.nan}),
            'annual_returns_5yr': pd.Series({'SPY': np.nan}),
            'vola': pd.Series({'SPY': 0.1}),
            'ds_vola': pd.Series({'SPY': 0.08}),
            'std_dev_1yr': pd.Series({'SPY': 0.11}),
            'std_dev_3yr': pd.Series({'SPY': 0.12}),
            'std_dev_5yr': pd.Series({'SPY': 0.13}),
        }
        out = format_investment_options_lines(
            ['"SPY","S&P 500","US Stocks",""'],
            metrics,
        )
        self.assertIn('"3.00","3.00"', out[0])

    @patch('brownbear.investment_options.compute_investment_metrics')
    @patch('brownbear.investment_options.compile_timeseries')
    @patch('brownbear.investment_options.fetch_timeseries')
    def test_update_investment_options_writes_output(
        self, mock_fetch, mock_compile, mock_metrics,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path = directory / 'investment-options-in.csv'
            input_path.write_text(
                '# comment\n'
                '"Investment Option","Description","Asset Class","3 mo","6 mo","1 Yr","1-1 Yr","3 Yr","5 Yr","Vola","DS Vola","SD 1 Yr","SD 3 Yr","SD 5 Yr"\n'
                '"SPY","S&P 500","US Stocks","","","","","","","","","",""\n'
            )

            dates = pd.date_range('2024-01-01', periods=260, freq='B')
            prices = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
            timeseries = pd.DataFrame({'SPY': prices})
            timeseries.index.name = 'Date'

            def write_timeseries(symbols, output_path='symbols-timeseries.csv'):
                timeseries.to_csv(output_path)

            mock_compile.side_effect = write_timeseries
            mock_metrics.return_value = {
                'annual_returns_3mo': pd.Series({'SPY': 1.0}),
                'annual_returns_6mo': pd.Series({'SPY': 2.0}),
                'annual_returns_1yr': pd.Series({'SPY': 3.0}),
                'annual_returns_1_1yr': pd.Series({'SPY': 4.0}),
                'annual_returns_3yr': pd.Series({'SPY': 5.0}),
                'annual_returns_5yr': pd.Series({'SPY': 6.0}),
                'vola': pd.Series({'SPY': 0.1}),
                'ds_vola': pd.Series({'SPY': 0.08}),
                'std_dev_1yr': pd.Series({'SPY': 0.11}),
                'std_dev_3yr': pd.Series({'SPY': 0.12}),
                'std_dev_5yr': pd.Series({'SPY': 0.13}),
            }

            result = update_investment_options(directory=directory, refresh_timeseries=False)

            mock_fetch.assert_called_once()
            mock_compile.assert_called_once()
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]['Investment Option'], 'SPY')
            self.assertTrue((directory / 'investment-options.csv').is_file())
            self.assertNotEqual(result.iloc[0]['1 Yr'], '')


if __name__ == '__main__':
    unittest.main()

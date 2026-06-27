import unittest
from unittest.mock import patch

import brownbear as bb


class TestComparePortfolios(unittest.TestCase):
    def test_matching_portfolios(self):
        portfolio = {'SPY': 10, 'BIL': 5}
        df = bb.compare_portfolios(portfolio, portfolio)
        self.assertTrue((df['delta'] == 0).all())

    def test_differences(self):
        current = {'SPY': 10, 'BIL': 5}
        target = {'SPY': 12, 'QQQ': 3}
        df = bb.compare_portfolios(current, target)
        by_symbol = df.set_index('symbol')
        self.assertEqual(by_symbol.loc['SPY', 'delta'], -2)
        self.assertEqual(by_symbol.loc['BIL', 'delta'], 5)
        self.assertEqual(by_symbol.loc['QQQ', 'delta'], -3)


class TestVerifyPortfolio(unittest.TestCase):
    @patch('brownbear.trade.fetch_schwab_portfolio')
    @patch('brownbear.trade.time.sleep')
    def test_preview_single_check(self, mock_sleep, mock_fetch):
        mock_fetch.return_value = {'SPY': 10}
        target = {'SPY': 10}

        comparison = bb.verify_portfolio('350', target, execute_live_trades=False)

        mock_fetch.assert_called_once()
        mock_sleep.assert_not_called()
        self.assertTrue((comparison['delta'] == 0).all())

    @patch('brownbear.trade.fetch_schwab_portfolio')
    @patch('brownbear.trade.time.sleep')
    def test_live_rechecks_until_match(self, mock_sleep, mock_fetch):
        mock_fetch.side_effect = [
            {'SPY': 8},
            {'SPY': 9},
            {'SPY': 10},
        ]
        target = {'SPY': 10}

        comparison = bb.verify_portfolio('350', target, execute_live_trades=True)

        self.assertEqual(mock_fetch.call_count, 3)
        mock_sleep.assert_any_call(10)
        mock_sleep.assert_any_call(20)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertTrue((comparison['delta'] == 0).all())

    @patch('brownbear.trade.fetch_schwab_portfolio')
    @patch('brownbear.trade.time.sleep')
    def test_live_still_differs_after_all_waits(self, mock_sleep, mock_fetch):
        mock_fetch.return_value = {'SPY': 8}
        target = {'SPY': 10}

        comparison = bb.verify_portfolio(
            '350',
            target,
            execute_live_trades=True,
            live_recheck_waits=(1, 2),
        )

        self.assertEqual(mock_fetch.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(comparison.loc[comparison['symbol'] == 'SPY', 'delta'].iloc[0], -2)


if __name__ == '__main__':
    unittest.main()

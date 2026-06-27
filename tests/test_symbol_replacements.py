import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import brownbear as bb
import symbol_replacements as sr


class TestSymbolReplacements(unittest.TestCase):
    def setUp(self):
        sr.load_replacements.cache_clear()

    def test_resolve_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'symbol-replacements.csv'
            path.write_text(
                'old_symbol,new_symbol,note,effective_date\n'
                'SATS,ECHO,EchoStar ticker change,2026-06-01\n'
            )
            with patch.object(sr, 'REPLACEMENTS_FILE', path):
                sr.load_replacements.cache_clear()
                self.assertEqual(sr.resolve_symbol('SATS'), 'ECHO')
                self.assertEqual(sr.resolve_symbol('sats'), 'ECHO')
                self.assertEqual(sr.resolve_symbol('SPY'), 'SPY')

    def test_normalize_portfolio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'symbol-replacements.csv'
            path.write_text(
                'old_symbol,new_symbol,note,effective_date\n'
                'SATS,ECHO,EchoStar ticker change,2026-06-01\n'
            )
            with patch.object(sr, 'REPLACEMENTS_FILE', path):
                sr.load_replacements.cache_clear()
                portfolio = sr.normalize_portfolio({'SATS': 2, 'SPY': 10})
                self.assertEqual(portfolio, {'ECHO': 2, 'SPY': 10})

    def test_lookup_quote_price_uses_old_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'symbol-replacements.csv'
            path.write_text(
                'old_symbol,new_symbol,note,effective_date\n'
                'SATS,ECHO,EchoStar ticker change,2026-06-01\n'
            )
            with patch.object(sr, 'REPLACEMENTS_FILE', path):
                sr.load_replacements.cache_clear()
                price = sr.lookup_quote_price('ECHO', {'SATS': 104.07})
                self.assertEqual(price, 104.07)

    def test_rebalance_orders_csv_uses_replaced_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'symbol-replacements.csv'
            path.write_text(
                'old_symbol,new_symbol,note,effective_date\n'
                'SATS,ECHO,EchoStar ticker change,2026-06-01\n'
            )
            with patch.object(sr, 'REPLACEMENTS_FILE', path):
                sr.load_replacements.cache_clear()
                orders = {
                    'sell': {'SATS': 2},
                    'buy': {},
                }
                df = bb.rebalance_orders_to_dataframe(
                    orders,
                    account='336',
                    quote_per_investment={'SATS': 104.07},
                )
                self.assertEqual(df.iloc[0]['symbol'], 'ECHO')
                self.assertEqual(df.iloc[0]['price'], 104.07)

    def test_rebalance_portfolio_normalizes_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'symbol-replacements.csv'
            path.write_text(
                'old_symbol,new_symbol,note,effective_date\n'
                'SATS,ECHO,EchoStar ticker change,2026-06-01\n'
            )
            with patch.object(sr, 'REPLACEMENTS_FILE', path):
                sr.load_replacements.cache_clear()
                orders = bb.rebalance_portfolio(
                    current_portfolio={'ECHO': 4},
                    target_portfolio={'SATS': 2},
                )
                self.assertEqual(orders['sell'], {'ECHO': 2})
                self.assertEqual(orders['buy'], {})

    @patch('yfinance.Ticker')
    def test_get_quote_uses_replacement(self, mock_ticker):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'symbol-replacements.csv'
            path.write_text(
                'old_symbol,new_symbol,note,effective_date\n'
                'SATS,ECHO,EchoStar ticker change,2026-06-01\n'
            )
            mock_ticker.return_value.fast_info = {'last_price': 105.5}
            with patch.object(sr, 'REPLACEMENTS_FILE', path):
                sr.load_replacements.cache_clear()
                quotes = bb.get_quote(['SATS'])
                self.assertEqual(quotes['SATS'], 105.5)
                mock_ticker.assert_called_once_with('ECHO')


if __name__ == '__main__':
    unittest.main()

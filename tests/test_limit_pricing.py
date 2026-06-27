import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from limit_pricing import compute_limit_price, spread_pct


class TestLimitPricing(unittest.TestCase):
  def test_spread_pct(self):
    self.assertAlmostEqual(spread_pct(100, 100.5), 0.5 / 100.5)

  def test_aggressive_buy(self):
    result = compute_limit_price('BUY', 100, 100.5, 100.25, 'aggressive', 0.01)
    self.assertEqual(result.price, 100.5)
    self.assertEqual(result.strategy, 'aggressive')
    self.assertFalse(result.non_marketable)

  def test_aggressive_sell(self):
    result = compute_limit_price('SELL', 100, 100.5, 100.25, 'aggressive', 0.01)
    self.assertEqual(result.price, 100)

  def test_pennying_buy(self):
    result = compute_limit_price('BUY', 100, 100.5, 100.25, 'pennying', 0.01)
    self.assertEqual(result.price, 100.01)

  def test_pennying_sell(self):
    result = compute_limit_price('SELL', 100, 100.5, 100.25, 'pennying', 0.01)
    self.assertEqual(result.price, 100.49)

  def test_midpoint(self):
    result = compute_limit_price('BUY', 100, 102, 101, 'midpoint', 0.05)
    self.assertEqual(result.price, 101)

  def test_wide_spread_fallback(self):
    result = compute_limit_price('BUY', 100, 110, 105, 'aggressive', 0.005)
    self.assertEqual(result.price, 105)
    self.assertEqual(result.strategy, 'wide_spread_fallback')
    self.assertFalse(result.non_marketable)


if __name__ == '__main__':
  unittest.main()

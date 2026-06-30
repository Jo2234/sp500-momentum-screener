from datetime import date
import json

import pandas as pd
import pytest

from sp500_momentum import strategy
from sp500_momentum.backtester import (
    apply_round_trip_costs,
    calculate_holding_period_end,
    calculate_portfolio_return,
    create_backtest_manifest,
    run_backtest,
)


def price_frame(start, periods, first=100.0, last=121.0):
    idx = pd.bdate_range(start=start, periods=periods)
    values = pd.Series([first] + [first] * (periods - 2) + [last], index=idx)
    return pd.DataFrame({"Close": values})


def test_calculate_annual_return_uses_first_and_last_prices_in_year():
    prices = pd.concat([
        price_frame("2022-12-15", 12, 90, 99),
        price_frame("2023-01-03", 20, 100, 121),
        price_frame("2024-01-03", 12, 121, 130),
    ])

    assert strategy.calculate_annual_return(prices, 2023) == pytest.approx(0.21)


def test_calculate_annual_return_requires_enough_trading_days():
    prices = price_frame("2023-01-03", 9, 100, 110)

    assert strategy.calculate_annual_return(prices, 2023) is None


def test_run_momentum_screen_uses_prior_year_window_without_lookahead(monkeypatch):
    calls = {}

    def fake_constituents(as_of):
        calls["constituents_date"] = as_of
        return ["AAA", "BBB"]

    def fake_fetch_batch(tickers, start_date, end_date, workers=10, show_progress=True):
        calls["stock_window"] = (list(tickers), start_date, end_date)
        frame = pd.concat([
            price_frame("2020-01-02", 20, 100, 130),
            price_frame("2021-01-04", 20, 130, 169),
        ])
        return {ticker: frame for ticker in tickers}

    def fake_sp500(start_date, end_date):
        calls["benchmark_window"] = (start_date, end_date)
        return pd.concat([
            price_frame("2020-01-02", 20, 100, 105),
            price_frame("2021-01-04", 20, 105, 110),
        ])

    monkeypatch.setattr(strategy, "get_constituents_at_date", fake_constituents)
    monkeypatch.setattr(strategy, "fetch_batch_data", fake_fetch_batch)
    monkeypatch.setattr(strategy, "get_sp500_index_data", fake_sp500)

    top, benchmark, summary = strategy.run_momentum_screen(
        analysis_date=date(2022, 6, 15),
        lookback_years=2,
        top_n=1,
        show_progress=False,
    )

    assert calls["constituents_date"] == date(2020, 1, 1)
    assert calls["stock_window"] == (["AAA", "BBB"], date(2020, 1, 1), date(2021, 12, 31))
    assert calls["benchmark_window"] == (date(2020, 1, 1), date(2021, 12, 31))
    assert top[0][0] == "AAA"
    assert set(benchmark) == {2020, 2021}
    assert not summary.empty


def test_backtest_holding_window_starts_on_strategy_date_and_ends_after_period(monkeypatch):
    import sp500_momentum.backtester as backtester

    calls = {}

    def fake_screen(analysis_date, lookback_years, top_n, workers, show_progress):
        calls["analysis_date"] = analysis_date
        return [("AAA", 0.5, {2023: 0.5})], {2023: 0.1}, pd.DataFrame({"Ticker": ["AAA"]})

    def fake_fetch(tickers, start_date, end_date, workers=10, show_progress=False):
        calls["portfolio_window"] = (list(tickers), start_date, end_date)
        return {"AAA": price_frame("2024-01-02", 20, 100, 120)}

    def fake_sp500(start_date, end_date):
        calls["benchmark_holding_window"] = (start_date, end_date)
        return price_frame("2024-01-02", 20, 100, 110)

    monkeypatch.setattr(backtester, "run_momentum_screen", fake_screen)
    monkeypatch.setattr(backtester, "fetch_batch_data", fake_fetch)
    monkeypatch.setattr(backtester, "get_sp500_index_data", fake_sp500)

    result = run_backtest(
        strategy_start_date=date(2024, 1, 1),
        lookback_years=1,
        top_n=1,
        holding_period_months=12,
        show_progress=False,
    )

    assert calls["analysis_date"] == date(2024, 1, 1)
    assert calls["portfolio_window"] == (["AAA"], date(2024, 1, 1), date(2025, 1, 1))
    assert calls["benchmark_holding_window"] == (date(2024, 1, 1), date(2025, 1, 1))
    assert result["portfolio_return"] == pytest.approx(0.20)
    assert result["benchmark_return"] == pytest.approx(0.10)


def test_calculate_holding_period_end_handles_months_and_day_28_cap():
    assert calculate_holding_period_end(date(2024, 1, 31), 12) == date(2025, 1, 28)
    assert calculate_holding_period_end(date(2024, 11, 15), 3) == date(2025, 2, 15)


def test_transaction_costs_and_slippage_are_applied_per_side(monkeypatch):
    import sp500_momentum.backtester as backtester

    gross = 0.20
    expected = (1 - 0.0015) * (1 + gross) * (1 - 0.0015) - 1
    assert apply_round_trip_costs(gross, transaction_cost_bps=10, slippage_bps=5) == pytest.approx(expected)

    def fake_fetch(tickers, start_date, end_date, workers=10, show_progress=False):
        return {"AAA": price_frame("2024-01-02", 20, 100, 120)}

    monkeypatch.setattr(backtester, "fetch_batch_data", fake_fetch)
    portfolio_return, individual = calculate_portfolio_return(
        ["AAA"], date(2024, 1, 1), date(2025, 1, 1), transaction_cost_bps=10, slippage_bps=5
    )

    assert individual["AAA"] == pytest.approx(expected)
    assert portfolio_return == pytest.approx(expected)


def test_backtest_manifest_is_deterministic_and_hashes_outputs(tmp_path, monkeypatch):
    import sp500_momentum.backtester as backtester

    output = tmp_path / "result.csv"
    output.write_text("Ticker,Return\nAAA,0.2\n", encoding="utf-8")
    monkeypatch.setattr(backtester, "_git_commit", lambda: "abc123")

    result = {
        "start_date": date(2024, 1, 1),
        "end_date": date(2025, 1, 1),
        "year": 2024,
        "selected_stocks": ["AAA"],
        "portfolio_return": 0.20000000000004,
        "benchmark_return": 0.1,
        "alpha": 0.10000000000004,
        "individual_returns": {"AAA": 0.20000000000004},
        "selection_returns": {"AAA": {2023: 0.4}},
    }
    params = {"mode": "backtest", "transaction_cost_bps": 10, "slippage_bps": 5}

    first = create_backtest_manifest(result, params, [output], command="sp500-momentum ...")
    second = create_backtest_manifest(result, params, [output], command="sp500-momentum ...")

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["outputs"][0]["sha256"]
    assert first["result"]["portfolio_return"] == 0.2

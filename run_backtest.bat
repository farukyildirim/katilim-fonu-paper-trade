@echo off
cd /d C:\Users\PC\PycharmProjects\multi_asset_ai_portfolio
python tefas_ai_fund.py --backtest --kind YAT --freq 7 --window 60 --top 3 --days 730 > backtest_top3.log 2>&1
echo DONE >> backtest_top3.log
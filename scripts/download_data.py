import yfinance as yf

def download_data(tickers, start_date, end_date, save_path):
    adj_close_df = yf.download(tickers, start = start_date, end = end_date)["Close"]
    adj_close_df.to_csv(save_path)

if __name__ == "__main__":
    banks = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "COF", "BK", "SCHW"]
    tech = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "INTC", "TXN"]
    retail = ["WMT", "TGT", "COST", "HD", "LOW", "TJX", "ROST", "KR", "DG", "DLTR", "BBY"]
    airlines = ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "ALGT", "SKYW"]

    tickers = banks + tech + retail + airlines
    start_date = "2017-01-01"
    end_date = "2025-01-01"
    save_path = "data/raw/adj_close_prices.csv"

    download_data(tickers, start_date, end_date, save_path)
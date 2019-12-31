import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from clint.textui import progress
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from struct import unpack
from collections import namedtuple, Counter
from datetime import timedelta
from time import time


def maybe_download(url):
    """Download and unzip ITCH data if not yet available"""
    filename = data_path / url.split("/")[-1]
    if not data_path.exists():
        print("Creating directory")
        data_path.mkdir()
    if not filename.exists():
        print("Downloading...", url)
        urlretrieve(url, filename)
    unzipped = data_path / (filename.stem + ".bin")
    if not unzipped.exists():
        print("Unzipping to", unzipped)
        with gzip.open(str(filename), "rb") as f_in:
            with open(unzipped, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unzipped


def clean_message_types(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df.value = df.value.str.strip()
    df.name = (
        df.name.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    df.notes = df.notes.str.strip()
    df["message_type"] = df.loc[df.name == "message_type", "value"]
    return df


def format_alpha(mtype, data):
    """Process byte strings of type alpha"""
    for col in alpha_formats.get(mtype).keys():
        if mtype != "R" and col == "stock":
            data = data.drop(col, axis=1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(encoding.get(col))
    return data


def process_itch_file(filename):
    messages = {}
    message_count = 0
    start = time()
    with filename.open("rb") as data:
        while True:
            # determine message size in bytes
            message_size = int.from_bytes(data.read(2), byteorder="big", signed=False)

            # get message type by reading first byte
            message_type = data.read(1).decode("ascii")

            # create data structure to capture result
            if not messages.get(message_type):
                messages[message_type] = []

            # read and store message
            record = data.read(message_size - 1)
            message = message_fields[message_type]._make(
                unpack(fstring[message_type], record)
            )
            messages[message_type].append(message)

            # deal with system events
            if message_type == "S":
                timestamp = int.from_bytes(message.timestamp, byteorder="big")
                print(
                    "\n", event_codes.get(message.event_code.decode("ascii"), "Error")
                )
                print(
                    "\t{0}\t{1:,.0f}".format(
                        timedelta(seconds=timestamp * 1e-9), message_count
                    )
                )
                if message.event_code.decode("ascii") == "C":
                    store_messages(messages)
                    break

            message_count += 1
            if message_count % 2.5e7 == 0:
                timestamp = int.from_bytes(message.timestamp, byteorder="big")
                print(
                    "\t{0}\t{1:,.0f}\t{2}".format(
                        timedelta(seconds=timestamp * 1e-9),
                        message_count,
                        timedelta(seconds=time() - start),
                    )
                )
                store_messages(messages)
                messages = {}

    print(timedelta(seconds=time() - start))


def store_messages(m):
    """Handle occasional storing of all messages"""
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items():
            # convert to DataFrame
            data = pd.DataFrame(data)

            # parse timestamp info
            data.timestamp = data.timestamp.apply(int.from_bytes, byteorder="big")
            data.timestamp = pd.to_timedelta(data.timestamp)

            # apply alpha formatting
            if mtype in alpha_formats.keys():
                data = format_alpha(mtype, data)

            s = alpha_length.get(mtype)
            if s:
                s = {c: s.get(c) for c in data.columns}
            dc = ["stock_locate"]
            if m == "R":
                dc.append("stock")
            store.append(mtype, data, format="t", min_itemsize=s, data_columns=dc)


def dump_trading_messages_by_frequency():
    """Trading Message Frequency"""
    message_type_counter = Counter()

    with pd.HDFStore(itch_store) as store:
        keys = store.keys()
        keys.remove("/summary")
        for k in keys:
            print(f"Reading {k}")
            df = store[k]
            message_type_counter[k[1:]] = len(df.index)

    counter = pd.Series(message_type_counter).to_frame("# Trades")
    counter["Message Type"] = counter.index.map(
        message_labels.set_index("message_type").name.to_dict()
    )
    counter = counter[["Message Type", "# Trades"]].sort_values(
        "# Trades", ascending=False
    )
    print(counter)

    with pd.HDFStore(itch_store) as store:
        store.put("summary", counter)


def plot_top_equities_by_value():
    # Top Equities by Traded Value
    with pd.HDFStore(itch_store) as store:
        stocks = store["R"].loc[:, ["stock_locate", "stock"]]
        trades = (
            store["P"]
            .append(store["Q"].rename(columns={"cross_price": "price"}), sort=False)
            .merge(stocks)
        )
    trades["value"] = trades.shares.mul(trades.price)
    trades["value_share"] = trades.value.div(trades.value.sum())
    trade_summary = (
        trades.groupby("stock").value_share.sum().sort_values(ascending=False)
    )
    trade_summary.iloc[:50].plot.bar(
        figsize=(14, 6), color="darkblue", title="Share of Traded Value"
    )
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    plt.savefig(plots_path / "equities_by_value.png")


def get_stock_messages(date, stock):
    """Collect trading messages for given stock"""
    with pd.HDFStore(itch_store) as store:
        stock_locate = store.select("R", where="stock = stock").stock_locate.iloc[0]
        target = "stock_locate = stock_locate"

        data = {}
        # trading messsage types
        messages = ["A", "F", "E", "C", "X", "D", "U", "P", "Q"]
        for m in messages:
            data[m] = (
                store.select(m, where=target)
                .drop("stock_locate", axis=1)
                .assign(type=m)
            )

    order_cols = ["order_reference_number", "buy_sell_indicator", "shares", "price"]
    orders = pd.concat([data["A"], data["F"]], sort=False, ignore_index=True).loc[
        :, order_cols
    ]

    for m in messages[2:-3]:
        data[m] = data[m].merge(orders, how="left")

    data["U"] = data["U"].merge(
        orders,
        how="left",
        right_on="order_reference_number",
        left_on="original_order_reference_number",
        suffixes=["", "_replaced"],
    )

    data["Q"].rename(columns={"cross_price": "price"}, inplace=True)
    data["X"]["shares"] = data["X"]["cancelled_shares"]
    data["X"] = data["X"].dropna(subset=["price"])

    data = pd.concat([data[m] for m in messages], ignore_index=True, sort=False)
    data["date"] = pd.to_datetime(date, format="%m%d%Y")
    data.timestamp = data["date"].add(data.timestamp)
    data = data[data.printable != 0]

    drop_cols = [
        "tracking_number",
        "order_reference_number",
        "original_order_reference_number",
        "cross_type",
        "new_order_reference_number",
        "attribution",
        "match_number",
        "printable",
        "date",
        "cancelled_shares",
    ]
    return data.drop(drop_cols, axis=1).sort_values("timestamp").reset_index(drop=True)


def get_trades(m):
    """Combine C, E, P and Q messages into trading records"""
    trade_dict = {"executed_shares": "shares", "execution_price": "price"}
    cols = ["timestamp", "executed_shares"]
    trades = (
        pd.concat(
            [
                m.loc[m.type == "E", cols + ["price"]].rename(columns=trade_dict),
                m.loc[m.type == "C", cols + ["execution_price"]].rename(
                    columns=trade_dict
                ),
                m.loc[m.type == "P", ["timestamp", "price", "shares"]],
                m.loc[m.type == "Q", ["timestamp", "price", "shares"]].assign(cross=1),
            ],
            sort=False,
        )
        .dropna(subset=["price"])
        .fillna(0)
    )
    return trades.set_index("timestamp").sort_index().astype(int)


def add_orders(orders, buysell, nlevels):
    """Add orders up to desired depth given by nlevels;
    sell in ascending, buy in descending order"""
    new_order = []
    items = sorted(orders.copy().items())
    if buysell == 1:
        items = reversed(items)
    for i, (p, s) in enumerate(items, 1):
        new_order.append((p, s))
        if i == nlevels:
            break
    return orders, new_order


def build_order_book(order_book_store, messages):
    with pd.HDFStore(order_book_store) as store:
        key = f"{stock}/messages"
        store.put(key, messages)
        print(store.info())

    # Combine Trading Records
    trades = get_trades(messages)
    print(trades.info())

    with pd.HDFStore(order_book_store) as store:
        store.put(f"{stock}/trades", trades)

    # Create Orders
    order_book = {-1: {}, 1: {}}
    current_orders = {-1: Counter(), 1: Counter()}
    message_counter = Counter()
    nlevels = 100

    start = time()
    for message in messages.itertuples():
        i = message[0]
        if i % 1e5 == 0 and i > 0:
            print("{:,.0f}\t\t{}".format(i, timedelta(seconds=time() - start)))
            save_orders(order_book_store, order_book, append=True)
            order_book = {-1: {}, 1: {}}
            start = time()
        if np.isnan(message.buy_sell_indicator):
            continue
        message_counter.update(message.type)

        buysell = message.buy_sell_indicator
        price, shares = None, None

        if message.type in ["A", "F", "U"]:
            price = int(message.price)
            shares = int(message.shares)

            current_orders[buysell].update({price: shares})
            current_orders[buysell], new_order = add_orders(
                current_orders[buysell], buysell, nlevels
            )
            order_book[buysell][message.timestamp] = new_order

        if message.type in ["E", "C", "X", "D", "U"]:
            if message.type == "U":
                if not np.isnan(message.shares_replaced):
                    price = int(message.price_replaced)
                    shares = -int(message.shares_replaced)
            else:
                if not np.isnan(message.price):
                    price = int(message.price)
                    shares = -int(message.shares)

            if price is not None:
                current_orders[buysell].update({price: shares})
                if current_orders[buysell][price] <= 0:
                    current_orders[buysell].pop(price)
                current_orders[buysell], new_order = add_orders(
                    current_orders[buysell], buysell, nlevels
                )
                order_book[buysell][message.timestamp] = new_order

    message_counter = pd.Series(message_counter)
    print(message_counter)

    with pd.HDFStore(order_book_store) as store:
        print(store.info())


def save_orders(order_book_store, orders, append=False):
    cols = ["price", "shares"]
    for buysell, book in orders.items():
        df = pd.concat(
            [
                pd.DataFrame(data=data, columns=cols).assign(timestamp=t)
                for t, data in book.items()
            ]
        )
        key = "{}/{}".format(stock, order_dict[buysell])
        df.loc[:, ["price", "shares"]] = df.loc[:, ["price", "shares"]].astype(int)
        with pd.HDFStore(order_book_store) as store:
            if append:
                store.append(key, df.set_index("timestamp"), format="t")
            else:
                store.put(key, df.set_index("timestamp"))


data_path = Path("data")
itch_store = str(data_path / "itch.h5")

plots_path = Path("plots")

FTP_URL = "ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/"
SOURCE_FILE = "03272019.NASDAQ_ITCH50.gz"

filename = maybe_download(urljoin(FTP_URL, SOURCE_FILE))
date = filename.name.split(".")[0]

event_codes = {
    "O": "Start of Messages",
    "S": "Start of System Hours",
    "Q": "Start of Market Hours",
    "M": "End of Market Hours",
    "E": "End of System Hours",
    "C": "End of Messages",
}

encoding = {
    "primary_market_maker": {"Y": 1, "N": 0},
    "printable": {"Y": 1, "N": 0},
    "buy_sell_indicator": {"B": 1, "S": -1},
    "cross_type": {"O": 0, "C": 1, "H": 2},
    "imbalance_direction": {"B": 0, "S": 1, "N": 0, "O": -1},
}

formats = {
    ("integer", 2): "H",
    ("integer", 4): "I",
    ("integer", 6): "6s",
    ("integer", 8): "Q",
    ("alpha", 1): "s",
    ("alpha", 2): "2s",
    ("alpha", 4): "4s",
    ("alpha", 8): "8s",
    ("price_4", 4): "I",
    ("price_8", 8): "Q",
}

message_data = (
    pd.read_excel("message_types.xlsx", sheet_name="messages", encoding="latin1")
    .sort_values("id")
    .drop("id", axis=1)
)
message_types = clean_message_types(message_data)
message_labels = (
    message_types.loc[:, ["message_type", "notes"]]
    .dropna()
    .rename(columns={"notes": "name"})
)
message_labels.name = (
    message_labels.name.str.lower()
    .str.replace("message", "")
    .str.replace(".", "")
    .str.strip()
    .str.replace(" ", "_")
)

message_types.message_type = message_types.message_type.ffill()
message_types = message_types[message_types.name != "message_type"]
message_types.value = (
    message_types.value.str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)
message_types.info()

# TODO use these
# message_labels.to_csv('message_labels.csv', index=False)
# message_types = pd.read_csv('message_types.csv')

# print(message_labels.head())

message_types.loc[:, "formats"] = (
    message_types[["value", "length"]].apply(tuple, axis=1).map(formats)
)

alpha_fields = message_types[message_types.value == "alpha"].set_index("name")
alpha_msgs = alpha_fields.groupby("message_type")
alpha_formats = {k: v.to_dict() for k, v in alpha_msgs.formats}
alpha_length = {k: v.add(5).to_dict() for k, v in alpha_msgs.length}

message_fields, fstring = {}, {}
for t, message in message_types.groupby("message_type"):
    message_fields[t] = namedtuple(typename=t, field_names=message.name.tolist())
    fstring[t] = ">" + "".join(message.formats.tolist())

print(f"Looking for {itch_store}")
if not Path(itch_store).exists():
    print("No ITCH Store")
    process_itch_file(filename)
else:
    print("Found ITCH Store")

# Summarize Trading Day
# dump_trading_messages_by_frequency()
# plot_top_equities_by_value()

# Build Order Book
order_dict = {-1: "sell", 1: "buy"}

# TODO store in single order book or separate?
stock = "AAPL"
order_book_store = data_path / f"{stock.lower()}_order_book.h5"

if not Path(order_book_store).exists():
    messages = get_stock_messages(date=date, stock=stock)
    messages.info(null_counts=True)
    build_order_book(order_book_store, messages)
else:
    print("Found Order Book Store")

with pd.HDFStore(order_book_store) as store:
    buy = store[f"{stock}/buy"].reset_index().drop_duplicates()
    sell = store[f"{stock}/sell"].reset_index().drop_duplicates()

# Price to decimals
buy.price = buy.price.mul(1e-4)
sell.price = sell.price.mul(1e-4)

# Remove outliers
percentiles = [.01, .02, .1, .25, .75, .9, .98, .99]
pd.concat([buy.price.describe(percentiles=percentiles).to_frame('buy'),
           sell.price.describe(percentiles=percentiles).to_frame('sell')], axis=1)

buy = buy[buy.price > buy.price.quantile(.01)]
sell = sell[sell.price < sell.price.quantile(.99)]

# Buy-Sell Order Distrubution
market_open = '0930'
market_close = '1600'

fig, ax = plt.subplots(figsize=(7,5))
hist_kws = {'linewidth': 1, 'alpha': .5}
sns.distplot(buy.set_index('timestamp').between_time(market_open, market_close).price, ax=ax, label='Buy', kde=False, hist_kws=hist_kws)
sns.distplot(sell.set_index('timestamp').between_time(market_open, market_close).price, ax=ax, label='Sell', kde=False, hist_kws=hist_kws)
plt.legend(fontsize=10)
plt.title('Limit Order Price Distribution', fontsize=14)
ax.set_yticklabels(['{:,}'.format(int(y/1000)) for y in ax.get_yticks().tolist()])
ax.set_xticklabels(['${:,}'.format(int(x)) for x in ax.get_xticks().tolist()])
plt.xlabel('Price', fontsize=12)
plt.ylabel('Shares (\'000)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/price_distribution', dpi=600)

# Order Book Depth
utc_offset = timedelta(hours=4)
depth = 100

buy_per_min = (buy
               .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])
               .shares
               .sum()
               .apply(np.log)
               .to_frame('shares')
               .reset_index("price")
               .between_time(market_open, market_close)
               .groupby(level='timestamp', as_index=False, group_keys=False)
               .apply(lambda x: x.nlargest(columns='price', n=depth))
               .reset_index())
buy_per_min.timestamp = buy_per_min.timestamp.add(utc_offset).astype(int)
buy_per_min.info()

sell_per_min = (sell
                .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])
                .shares
                .sum()
                .apply(np.log)
                .to_frame('shares')
                .reset_index('price')
                .between_time(market_open, market_close)
                .groupby(level='timestamp', as_index=False, group_keys=False)
                .apply(lambda x: x.nsmallest(columns='price', n=depth))
                .reset_index())
sell_per_min.timestamp = sell_per_min.timestamp.add(utc_offset).astype(int)
sell_per_min.info()

with pd.HDFStore(order_book_store) as store:
    trades = store[f"{stock}/trades"]
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close)

trades_per_min = (trades
                  .resample('Min')
                  .agg({'price': 'mean', 'shares': 'sum'}))
trades_per_min.index = trades_per_min.index.to_series().add(utc_offset).astype(int)
trades_per_min.info()

fig, ax = plt.subplots(figsize=(7, 5))

buy_per_min.plot.scatter(x='timestamp', y='price', c='shares', ax=ax, colormap='Blues', colorbar=False, alpha=.25)
sell_per_min.plot.scatter(x='timestamp', y='price', c='shares', ax=ax, colormap='Reds', colorbar=False, alpha=.25)
trades_per_min.price.plot(figsize=(14, 8), c='k', ax=ax, lw=2,
                          title=f'{stock} | {date} | Buy and Sell Limit Order Book | Depth = {depth}')

xticks = [datetime.fromtimestamp(ts / 1e9).strftime('%H:%M') for ts in ax.get_xticks()]
ax.set_xticklabels(xticks)
ax.set_xlabel('')
ax.set_ylabel('Price')

fig.tight_layout()
fig.savefig('plots/order_book2', dpi=600)

print("\nDone")

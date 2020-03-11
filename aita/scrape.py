from aita.aita_search import AitaSubmissionSearch
import asyncio
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

async def create_record_batch(data, fields):
    df = pd.DataFrame([d.d_ for d in data if int(d.d_['score']) >= 1])

    for f in fields:
        if f not in df.columns:
            df[f] = ''

    print(df['title'][0:9].values)
    print('\n')
    return pa.Table.from_pandas(df[fields]), df.columns.values


async def do_search(aita, start_date, end_date):
    for sub in aita.search(start_date, end_date, return_batch=True):
        yield await create_record_batch(sub, aita.fields)


async def main():
    with open('credentials.json') as creds_file:
        creds = json.load(creds_file)
    aita = AitaSubmissionSearch(credentials=creds, size=500)
    counter = 0
    search = do_search(aita, start_date='2010-01-01', end_date='2020-03-01')
    rows = 0
    async for tbl in search:

        tables.append(tbl)
        rows += tables[-1].shape[0]
        if rows > 100000:
            pq.write_table(pa.concat_tables(tables), f'data/pq/aita_{counter}.parquet')
            tables = []
            rows = 0
            counter += 1

if __name__ == '__main__':
    asyncio.run(main())


#!/usr/bin/env python
# -*- coding: utf-8; -*-

from datetime import date, datetime as dt

import psycopg2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.dates import HourLocator, DateFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

today = date.today()
conn = psycopg2.connect('postgresql://rpi@192.168.0.37:5432/homemetrics')

def dfday_fromsql(conn, date=today):
    params = {'date': date}
    sql = '''SELECT * FROM "Gas" WHERE date_trunc('day', ts) = %(date)s'''
    df = pd.read_sql(sql, conn, parse_dates=['ts'], index_col='ts', params=params)
    df['delta'] = df['cm3'] - df['cm3'].min()
    return df


#-----------------------------------------------------------------------------
df = dfday_fromsql(conn)
df = df[dt(2014,12,5,2,30):dt(2014,12,5,3,00)]
X = df.index
Y = df['delta']

plt.style.use('/home/gv/source/github/semi-smart-meter/src/plotstyle.rc')

#-----------------------------------------------------------------------------
figure = Figure(figsize=(12, 5))
canvas = FigureCanvas(figure)
ax = figure.add_subplot(111)
ax.plot(X, Y, '-')

ax.set_title('Daily Gas', size='x-small')
ax.set_xlabel('Time')
ax.set_ylabel(r'$m^{3}$')

ax.set_ylim(Y.min(), Y.max())

def yticks_formatter(val, n):
    if n == 0:
        return df['cm3'].min() // 100
    if n == ax.get_yticks().shape[0] - 2:
        return df['cm3'].max() // 100
    return '+%.2f' % (val / 100)

# def xticks_formatter(val, n):
#     import pdb; pdb.set_trace()
#     return val

ax.yaxis.set_major_locator(MaxNLocator())
ax.xaxis.set_major_locator(HourLocator())

ax.yaxis.set_major_formatter(FuncFormatter(yticks_formatter))
ax.xaxis.set_major_formatter(DateFormatter('%H'))

ax.grid(True)

canvas.print_figure('/tmp/test.png')
import subprocess
subprocess.check_call('sxiv /tmp/test.png', shell=True)



# from ephem import Observer, Sun
# eind = Observer()
# eind.date = '2014-12-05 00:00:01'
# eind.lat  = '51.44'
# eind.lon  = '5.47'
# eind.elev = 17
# eind.horizon = '-6'
# sunrise = eind.previous_rising(Sun(), use_center=True)
# sunset  = eind.next_setting   (Sun(), use_center=True)
# print('Sunrise: %s, Sunset: %s' % (sunrise, sunset))

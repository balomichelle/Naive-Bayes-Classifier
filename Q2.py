#/usr/bin/env python 3.6
# -*- coding: utf-8 -*-

import csv
import sqlite3
conn = sqlite3.connect('database1.db')
c = conn.cursor()

c.execute('CREATE TABLE testcsv0(obs INTEGER, TestRes INTEGER, Var2 INTEGER, Var3 INTEGER, Var4 INTEGER, Var5 INTEGER, Var6 INTEGER)')

with open ('Flying_Fitness.csv') as f:
    csv_data = csv.reader(f, delimiter= ',')
    next(csv_data,None) #skip the headers
    for row in csv_data:
        c.execute('INSERT INTO testcsv0 VALUES(?,?,?,?,?,?,?)', tuple(row))

c.execute('SELECT * from testcsv0')
print(*c.fetchall(), sep='\n')

conn.commit()
conn.close()

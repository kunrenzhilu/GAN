import records

db = records.Database('postgres://readonly:sapresearch@lssinh003.sin.sap.corp:5433/nse_2016_all')

rows = db.query('select e.nid, extract(doy from e.sgt)as day_of_year, e.triplabel as label, e.lat, e.lon'
                ' from allweeks_extra_2016 e, allweeks_tripsummary_2016 s '
                ' where s.school_loc[1] >= 1.385 and s.school_loc[1] <= 1.395'
                ' and s.school_loc[2] >= 103.865 and s.school_loc[2] <= 103.875'
                ' and s.nid = e.nid and lat != \'NaN\''' and lon != \'NaN\''
                ' and e.triplabel is not null and sgt is not null'
                ' order by e.nid, extract(doy from e.sgt), e.triplabel');
f = open('1.39_103.87.csv', 'w+')
f.write(rows.export('csv'))
f.close()
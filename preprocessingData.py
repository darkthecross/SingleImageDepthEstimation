from dbReader import dbReader

rb = dbReader()

print('loading')
rb.loadOrigin()

print('compressing')
rb.compress()

print('saving')
rb.saveCompressedData()

print('loading compressed')
rb.loadCompressed()

print(rb.img.shape)
print(rb.img.shape)

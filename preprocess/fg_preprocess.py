import h5py

cats_dict = {}
genders_dict = {}
brands_dict = {}
seasons_dict = {}
comps_dict = {}

# training 
all_data = h5py.File('fashiongen_256_256_train.h5', 'r')
brands = all_data['input_brand']
cats = all_data['input_subcategory']
genders = all_data['input_gender']
seasons = all_data['input_season']
comps = all_data['input_composition']


for i, b in enumerate(brands):
    b = b[0].decode('latin1')
    if b not in brands_dict:
        brands_dict[b] = 1
    c = cats[i][0].decode('latin1')
    if c not in cats_dict:
        cats_dict[c] = 1
    g = genders[i][0].decode('latin1')
    if g not in genders_dict:
        genders_dict[g] = 1

    s = seasons[i][0].decode('latin1')
    if s not in seasons_dict:
        seasons_dict[s] = 1

    p = comps[i][0].decode('latin1')
    if p not in comps_dict:
        comps_dict[p] = 1

    #ps = p.split(',')  # ['90% cotton', ' 8% polyester', ' 2% elastane.']
    #for m in ps:
    #    if '%' in m:
    #        m = m.split('%')[1].strip().strip('.')
    #    else:
    #        m = m.strip().strip('.')
    #    if m not in comps_dict:
    #        comps_dict[m] = 1




# validation 
all_data = h5py.File('fashiongen_256_256_validation.h5', 'r')
brands = all_data['input_brand']
cats = all_data['input_subcategory']
genders = all_data['input_gender']
seasons = all_data['input_season']
comps = all_data['input_composition']


for i, b in enumerate(brands):
    b = b[0].decode('latin1')
    if b not in brands_dict:
        brands_dict[b] = 1
        
    c = cats[i][0].decode('latin1')
    if c not in cats_dict:
        cats_dict[c] = 1
    g = genders[i][0].decode('latin1')
    if g not in genders_dict:
        genders_dict[g] = 1

    s = seasons[i][0].decode('latin1')
    if s not in seasons_dict:
        seasons_dict[s] = 1

    p = comps[i][0].decode('latin1')
    if p not in comps_dict:
        comps_dict[p] = 1

    #ps = p.split(',')  # ['90% cotton', ' 8% polyester', ' 2% elastane.']
    #for m in ps:   
    #    if '%' in m:
    #        m = m.split('%')[1].strip().strip('.')
    #    else:
    #        m = m.strip().strip('.')
    #    if m not in comps_dict:
    #        comps_dict[m] = 1


#
outf1 = open('brand_id.txt', 'w')
cc = 0
for k in brands_dict.keys():
    sss = "{},{}\n".format(k, cc)
    outf1.write(sss)
    cc += 1
outf1.close()
print('Total brands ', cc)

#
outf1 = open('cat_id.txt', 'w')
cc = 0
for k in cats_dict.keys():
    sss = "{},{}\n".format(k, cc)
    outf1.write(sss)
    cc += 1
outf1.close()
print('Total category ', cc)

# 
outf1 = open('gender_id.txt', 'w')
cc = 0
for k in genders_dict.keys():
    sss = "{},{}\n".format(k, cc)
    outf1.write(sss)
    cc += 1
outf1.close()
print('Total gender ', cc)



outf1 = open('season_id.txt', 'w')
cc = 0
for k in seasons_dict.keys():
    sss = "{},{}\n".format(k, cc)
    outf1.write(sss)
    cc += 1
outf1.close()
print('Total season ', cc)


outf1 = open('comps_id.txt', 'w')
cc = 0
for k in comps_dict.keys():
    sss = "{}&&{}\n".format(k, cc)
    outf1.write(sss)
    cc += 1
outf1.close()
print('Total materials ', cc)






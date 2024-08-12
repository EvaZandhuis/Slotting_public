

skus = 10

#number of skus per turnover rate
l_skus = int(0.2 * skus)
m_skus = int(0.2 * skus)
s_skus = int(0.6 * skus)

#percentage van totale voorraad
p_l = 0.8
p_m = 0.15
p_s = 0.05


#minimum amount is:  (for s_sku)
qb_s = 1


total_quantity = qb_s * s_skus / p_s


#quantity boxes per sku
qb_l = p_l *total_quantity /l_skus
qb_m = p_m *total_quantity /m_skus


print("total_quantity =", total_quantity)
print("number of boxes per l_skus =", qb_l)
print("number of boxes per m_skus =", qb_m)
print("number of boxes per s_skus =", qb_s)


#make dataframe with quantity boxes (D)
D_i = [qb_l] * l_skus + [qb_m] * m_skus + [qb_s] * s_skus
print(D_i)

k_systems = 5
tMax = 25000
Nexps = 1000

eps = 0.01
full_sync_counter = 0
not_sync_counter = 0
not_full_sync_indexes = []

final_time = time.time()
# random_IC = IC_random_generator(-3, 3)
R1_arr = []
R2_arr = []
IC_arr = []
index = 0
G_inh_arr = []


for i in range(0, Nexps, 1):
    print('Эксперимент ' + str(i+1))
    G_inh = 0.001

    random_IC = IC_random_generator(-3, 3)
    R1_arr_i, R2_arr_i, IC_i = make_investigation_of_the_dependence_of_the_order_parameters_on_the_initial_conditions(G_inh, random_IC)
    R1_arr.append(R1_arr_i)
    R2_arr.append(R2_arr_i)
    IC_arr.append(IC_i)
    G_inh_arr.append(G_inh)
    index+= 1
    print('-------------------------------------------------------------------')


print('Process end. Final time: ', time.time() - final_time)


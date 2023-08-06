import os
import random
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


s_values = ['1', '2', '3', '4', '5']
probabilities = [0.26, 0.22, 0.19, 0.18, 0.15]

wins_table = {('1', '1', '1'): 2,
            ('2', '2', '2'): 3,
            ('3', '3', '3'): 5,
            ('4', '4', '4'): 8,
            ('5', '5', '5'): 10}


num_threads = 1
num_spins = 300000
num_reels = 3
num_symbols = len(s_values)



def run_spins(num_spins, thread_num):
    rand_bytes = random.randbytes(num_spins * num_reels * 4)
    
    rand_bytes_uint32 = np.frombuffer(rand_bytes, dtype=np.uint32)
    cumulative_probabilities = np.cumsum(probabilities)
    reels = np.searchsorted(cumulative_probabilities, rand_bytes_uint32 / 2**32).reshape(num_spins, num_reels)
    values_array = np.array(s_values)[reels]

    # Calculate the total payout
    payout = np.zeros(num_spins//3)

    for i in range(0,num_spins,3):
        symbol = []
        symbol.append(tuple(values_array[i, :]))
        symbol.append(tuple(values_array[i+1, :]))
        symbol.append(tuple(values_array[i+2, :]))
        symbol.append(tuple( [values_array[i, 0],values_array[i+1, 0], values_array[i+2, 0]] ))
        symbol.append(tuple( [values_array[i, 1],values_array[i+1, 1], values_array[i+2, 1]] ))
        symbol.append(tuple( [values_array[i, 2],values_array[i+1, 2], values_array[i+2, 2]] ))
        symbol.append(tuple( [values_array[i, 0],values_array[i+1, 1], values_array[i+2, 2]] ))
        symbol.append(tuple( [values_array[i, 2],values_array[i+1, 1], values_array[i+2, 0]] ))
        
        for _ in symbol:
            if _ in wins_table:
                payout[i//3]+=wins_table[_]
        

    return payout


from tqdm.auto import tqdm

def rtp(num_spins):
    # Calculate the return to player (RTP) over a given number of spins
    payments = 0
    spins_per_thread = num_spins // num_threads

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_payouts = {executor.submit(run_spins, spins_per_thread, i): i for i in range(num_threads)}

        # Display progress bar
        progress_bar = tqdm(concurrent.futures.as_completed(future_payouts), total=num_threads, desc="Progress", unit="thread")

        for future in progress_bar:
            payments += future.result().sum()


    rtp = 3*payments / (num_spins)
    return rtp, future_payouts

def display_hits_percentage(payouts):
    c=0
    for t in payouts:
        if t !=0:
            c+=1

    print(f"Hits per spin=  {3*c / (num_spins):.2%}" )


rtp, future_payouts = rtp(num_spins)

# Get the payouts
payouts = np.concatenate([future.result() for future in future_payouts])

print(f"RTP: {rtp:.2%}")
display_hits_percentage( payouts)
print( "variance=", np.var(payouts[::3]))

# ваши требования не могут быть выполнены, минимум rtp примерно равен вообще 1.42
# простенькая минимизация
minn = 2
for i in range(40):
    for j in range(30):
        for k in range(j):
            for l in range(k):
                i1 = i/100
                j1 = j/100
                k1 = k/100
                l1 = l/100
                n1 = 1 - i1 - j1 - k1 - l1;
                d=(2*(i1**3) +3*(j1**3)+5*(k1**3)+8**(l1**3)+10*(n1**3)) 
                if d<minn:
                    minn = d
print("minimum of rtp= ", minn)
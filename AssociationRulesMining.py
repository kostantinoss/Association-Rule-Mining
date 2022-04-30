# -----------------------------
# Konstantinos Chondralis  3109
# -----------------------------

import pandas as pd
import numpy as np
import csv
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile, pstats, io
import draw_rules_graph as dg
from pstats import SortKey
from collections import defaultdict 
from itertools import combinations


path = ""
df = pd.read_csv(path + 'ratings_100users.csv')
num_of_movies = df["movieId"].nunique()
num_of_users = df["userId"].nunique()
min_rating = 3.5


def ReadMovies():
    return pd.read_csv(path + 'movies.csv')

def HashedCountersOfPairs():
    
    matrix = defaultdict(int)
    
    for basket in userBaskets:
        for i in userBaskets[basket]:
            for j in userBaskets[basket]:
                if i != j:
                    matrix[i, j] += 1
                    

    for key in matrix:
        print(key, matrix[key])
    
    return matrix


def CreateMovieBaskets(min_rating):

    userBaskets = defaultdict(list)
    outFile = open('userBaskets.csv', 'w')
    writer = csv.writer(outFile, delimiter=',')

    prev = 1
    for row in df.itertuples():
        userId = row[1]
        movieId = row[2]
        rating = row[3]
        
        if rating < min_rating:
            continue

        userBaskets[userId].append(movieId)

        if row[1] != prev:
            prev = row[1]

    for basket in userBaskets:
        userBaskets[basket] = sorted(userBaskets[basket])
        writer.writerow(userBaskets[basket])

    outFile.close()

    return userBaskets


def myApriori2(itemBaskets, min_frequency = 0.15, max_item_size = 4):

    frequentSingletons = set()
    frequentPairs = set()
    frequentK = defaultdict(set)
    frequentK[1] = frequentSingletons
    frequentK[2] = frequentPairs
    counterDict1 = defaultdict(int)
    N = len(itemBaskets)

    # ========================================================
    # 1st pass
    # ========================================================
    
    # Construct singletons
    # ---------------------
    for basket in itemBaskets:
        for item in itemBaskets[basket]:
            counterDict1[item] += 1
    
    # Filter frequent singletons
    # ---------------------------
    for item in counterDict1:
        frequency = counterDict1[item] / N
        if frequency >= min_frequency:
            frequentSingletons.add(item)


    # =========================================================
    # 2st pass
    # =========================================================
    
    # Construct pairs
    # ----------------
    for basket in itemBaskets:
        frequent_in_basket = set(itemBaskets[basket]).intersection(frequentSingletons)
        for i in combinations(frequent_in_basket, 2):
            i = tuple(sorted(i))
            counterDict1[i] += 1
            frequentPairs.add(i)
    
    # FIlter frequent pairs
    # ----------------------
    temp_frequentPairs = set()
    for pair in frequentPairs:
        frequency = counterDict1[tuple(pair)] / N
        if frequency >= min_frequency:
            temp_frequentPairs.add(tuple(pair))
    
    print(2, len(frequentPairs), len(temp_frequentPairs))
    frequentK[2] = temp_frequentPairs

    # ========================================================
    # k-th pass
    # ========================================================
    K = 3

    # Construct K-items
    #-------------------
    while True:

        if len(frequentK[K - 1]) == 0:
            break

        if K > max_item_size:
            break

        frequent_numbers_of_previous_kitems = [set( [ *i[0:] ] ) for i in frequentK[K - 1]]
        frequent_numbers_of_previous_kitems = set.union(*frequent_numbers_of_previous_kitems)

        for basket in itemBaskets:    
            frequent_in_basket = set(itemBaskets[basket]).intersection(frequent_numbers_of_previous_kitems)
            if len(frequent_in_basket) >= K:
                comb = combinations(frequent_in_basket, K)
                for i in comb:
                    i = tuple(sorted(i))
                    subset = set([ tuple(sorted(x)) for x in combinations(i, K - 1)])
                    if subset.issubset(frequentK[K - 1]):   
                        counterDict1[i] += 1
                        frequentK[K].add(i)

        # Filter K-items
        #----------------
        temp_frequent = set()
        for item in frequentK[K]:
            frequency = counterDict1[tuple(item)] / N
            if frequency >= min_frequency:
                temp_frequent.add(tuple(item))
        
        print(K, len(frequentK[K]), len(temp_frequent))
        frequentK[K] = temp_frequent
        K += 1

    return frequentK, counterDict1


def AssociationRulesCreation(frequentItems, itemsSupport, min_confidence = 0.5, MinLift = -1, MaxLift = -1):
    N = 100
    rule_id = 1
    columns = ['itemset', 'rule', 'hypothesis', 'conclusion', 'confidence', 'lift', 'interest', 'rule_ID']
    temp_list = []

    # Check if input parameters are valid
    # -----------------------------------
    if MinLift != -1 and MinLift < 0:
        print("\nBAD input!: MinLift\n ---> Acceptable values: -1 or MinLift > 0")
        print("Termination...\n")
        time.sleep(1)
        exit()
    
    if min_confidence < 0 or min_confidence > 1:
        print("\nBAD input!: min_confidence\n ---> Acceptable values: 0 < min_confidence < 1")
        print("Termination...\n")
        time.sleep(1)
        exit()
    
    if MinLift != -1 and MinLift < 0:
        print("\nBAD input!: MaxLift\n ---> Acceptable values: 0 < MaxLift < 1")
        print("Termination...\n")
        time.sleep(1)
        exit()

    # Rules creation
    #----------------------
    for k in frequentItems:
        for itemset in frequentItems[k]:
            
            # Singletons don't produce any rules
            if k >= 2:
                i = 1

                while k - i >= 1:
                    for hypothesis in combinations(itemset, k - i):
                        hypothesis = list(hypothesis)
                        conclusion = list( set(itemset).difference(set(hypothesis)) )
                        itemset = tuple(sorted(itemset))

                        if len(hypothesis) == 1:        
                            confidence = itemsSupport[itemset] / float(itemsSupport[hypothesis[0]])
                        else:
                            confidence = itemsSupport[itemset] / float(itemsSupport[tuple(sorted(hypothesis))])
                                                
                        if len(conclusion) == 1:
                            frequency_of_conclusion = itemsSupport[int(conclusion[0])] / N
                        else:
                            frequency_of_conclusion = itemsSupport[tuple(sorted(conclusion))]

                        lift = confidence/ frequency_of_conclusion
                        interest = confidence - frequency_of_conclusion

                        if confidence > min_confidence:
                            if MaxLift != -1:
                                if lift > MaxLift:
                                    continue
                            if MinLift != -1:    
                                if lift < MinLift:
                                    continue
                            
                            data = [[list(itemset), '{} ---> {}'.format(hypothesis, conclusion),
                                                                            hypothesis,
                                                                            conclusion,
                                                                            confidence,
                                                                            lift,
                                                                            interest,
                                                                            rule_id]]
                            temp_list.append(pd.DataFrame(data, columns=columns))
                            rule_id += 1
                    
                    i += 1

    if len(temp_list) == 0:
        print(" Found NO rules that meet the imposed criteria")
        exit()

    return pd.concat(temp_list)


def SampledApriori(reservoir):

    NumbeOfDistinctUsersSoFar = 0
    SetOfUsers = set()
    SampleOfBaskets = defaultdict(set)
    ratings_stream = pd.read_csv(path + 'ratings_100users_shuffled.csv')
    outFile = open('userBaskets.csv', 'w')
    writer = csv.writer(outFile, delimiter=',')

    print("\n\n >>> IMPORTANT NOTICE: Press Ctrl + C to stop the sampling routine manually")
    time.sleep(2)
    print("\n Starting ReservoirSampling in 1 sec...")
    time.sleep(1)

    try:
        print(" === STARTED ===")
        for index, row in ratings_stream.iterrows():
            current_user = int(row['userId'])
            current_movie = int(row['movieId'])
            rating = int(row['rating'])

            if rating < 3.5:
                continue

            if current_user not in SetOfUsers:
                SetOfUsers.add(current_user)

                if NumbeOfDistinctUsersSoFar < reservoir:
                    SampleOfBaskets[current_user].add(current_movie)

                if NumbeOfDistinctUsersSoFar >= reservoir:
                    j = random.randrange(NumbeOfDistinctUsersSoFar + 1)

                    if j < reservoir:

                        # Replace the j-th key from SampleOfBaskets dictionary
                        # with 'current_user' from the stream
                        #------------------------------------------------------
                        keys = list(SampleOfBaskets.keys())
                        SampleOfBaskets.pop(keys[j])
                        keys[j] = current_user
                        SampleOfBaskets[keys[j]].add(current_movie)
                
                NumbeOfDistinctUsersSoFar += 1
                
            if current_user in SampleOfBaskets:
                SampleOfBaskets[current_user] = SampleOfBaskets[current_user].union({current_movie})

    except KeyboardInterrupt:
        pass
        

    # Keep snapsot of baskets to a csv file
    #--------------------------------------
    for basket in SampleOfBaskets:
        SampleOfBaskets[basket] = sorted(SampleOfBaskets[basket])
        writer.writerow(SampleOfBaskets[basket])

    outFile.close()

    print("\n\n Sampling terminated.")
    print(" --- > RESULT: {} samples", len(SampleOfBaskets))

    time.sleep(1)

    return SampleOfBaskets


def presentRules(rules_df):

    global df
    movies_df = ReadMovies()
    rightWidth = 70
    leftWidth = 50

    while True:
        print("=" * 130)
        print("\n")
        print("(a) List ALL discovered rules".ljust(leftWidth) + "[format: a]".ljust(rightWidth))
        print("\n")
        print("(b) List all rules containing a BAG of movies".ljust(leftWidth) + "[format: in their <ITEMSET|HYPOTHESIS|CONCLUSION> b,<i,h,c>,<comma-sep. movie IDs>]".rjust(rightWidth))
        print("\n")
        print("(c) COMPARE rules with <CONFIDENCE,LIFT>".ljust(leftWidth) + "[format: c]".ljust(rightWidth))
        print("\n")
        print("(h) Print the HISTOGRAM of <CONFIDENCE|LIFT >".ljust(leftWidth) + "[format: h,<c,l >]".ljust(rightWidth))
        print("\n")
        print("(m) Show details of a MOVIE".ljust(leftWidth) + "[format: m,<movie ID>]".ljust(rightWidth))
        print("\n")
        print("(r) Show a particular RULE".ljust(leftWidth) + "[format: r,<rule ID>]".ljust(rightWidth))
        print("\n")
        print("(s) SORT rules by increasing <CONFIDENCE|LIFT >".ljust(leftWidth) + "[format: s,<c,l >]".ljust(rightWidth))
        print("\n")
        print("(v) VISUALIZATION of association rules (sorted by lift)".ljust(leftWidth) + "[format: v,<draw_choice: [c(ircular),r(andom),s(pring)]>,<num of rules to show>]".ljust(rightWidth))
        print("\n")
        print("(e) EXIT".ljust(leftWidth) +  "[format: e]".ljust(rightWidth))
        print("\n")
        print("=" * 130)
        
        instruction = input("Enter instruction: $ ")
        splited_instruction = instruction.split(",")

        if splited_instruction[0] == 'a':

            # Print ALL discovered rules
            # --------------------------
            print(rules_df.to_string())
            print("\n Press any key to continue ")


        elif splited_instruction[0] == 'b':
            
            # Print rules containing a Bag of movies
            # --------------------------------------
            if len(splited_instruction) < 3:
                print(" BAD Input ")
                time.sleep(1)
                continue
        
            attribute = splited_instruction[1]
            bag_of_movies = set( [int(i) for i in splited_instruction[2:]] )

            if attribute == 'h':
                temp_df = rules_df.loc[ rules_df['hypothesis'].apply(lambda x: True if bag_of_movies.issubset(set(x)) else False) ]
            elif attribute == 'i':
                temp_df = rules_df.loc[ rules_df['itemset'].apply(lambda x: True if bag_of_movies.issubset(set(x)) else False) ]   
            elif attribute == 'c':
                temp_df = rules_df.loc[ rules_df['conclusion'].apply(lambda x: True if bag_of_movies.issubset(set(x)) else False) ]
            else:
                print(" BAD Input ")
                time.sleep(1)
                continue

            print(temp_df.to_string())
            print("\n Press any key to continue ")


        elif splited_instruction[0] == 'c':
            
            # COMPARE rules with <CONFIDENCE,LIFT>
            # ------------------------------------
            data = rules_df[['confidence', 'lift']]
            
            ax = sns.lmplot(x="lift", y="confidence", data=data, fit_reg=True)
            plt.show()
            
            print("\n Press any key to continue ")


        elif splited_instruction[0] == 'h':
            
            # Print the HISTOGRAM of <CONFIDENCE|LIFT >
            # -----------------------------------------
            if len(splited_instruction) != 2:
                print(" BAD Input ")
                time.sleep(1)
                continue

            if splited_instruction[1] == 'c':
                attribute = 'confidence'
            
            elif splited_instruction[1] == 'l':
                attribute = 'lift'
            else:
                print(" BAD Input ")
                time.sleep(1)
                continue

            
            # ax = sns.distplot(rules_df[attribute], color='red', bins=12)
            rules_df[attribute].plot.hist(bins=12, alpha=0.5)
            
            plt.title('Histogram of CONFIDENCES among discovered rules')
            plt.ylabel("Number of ruels")
            plt.xlabel(attribute)
            plt.tight_layout() 
            plt.show()
            
            print("\n Press any key to continue ")


        elif splited_instruction[0] == 'm':
            
            # Show details of a MOVIE
            # -----------------------
            if len(splited_instruction) != 2:
                print(" BAD Input ")
                time.sleep(1)
                continue

            movie_ID = int(splited_instruction[1])
            temp_df = movies_df.loc[ movies_df['movieId'] == movie_ID]
            
            print(temp_df.to_string())
            print("\n Press any key to continue ")


        elif splited_instruction[0] == 'r':
            
            # Show a particular RULE
            # -----------------------
            if len(splited_instruction) != 2:
                print(" BAD Input ")
                time.sleep(1)
                continue

            ruleID = int(splited_instruction[1])
            temp_df = rules_df.loc[ rules_df['rule_ID'] == ruleID ]
            
            print(temp_df.to_string())
            print("\n Press any key to continue ")
            

        elif splited_instruction[0] == 's':
            
            # SORT rules by increasing <CONFIDENCE|LIFT >
            # -------------------------------------------
            
            if len(splited_instruction) != 3:
                print(" BAD Input ")
                time.sleep(1)
                continue

            if splited_instruction[1] == 'c':
                attribute = 'confidence'

            elif splited_instruction[1] == 'l':
                attribute = 'lift'

            else:
                print("BAD Input")
                time.sleep(1)
                continue

            print(rules_df.sort_values(by=attribute).to_string())
            print("\n Press any key to continue ")


        elif splited_instruction[0] == 'v':

            # Visualization of rules (sorted by lift)
            # ---------------------------------------
            draw_choice = splited_instruction[1]
            number_of_rules_to_draw = int(splited_instruction[2])

            data = rules_df.sort_values(by='lift')
            data = data.iloc[:number_of_rules_to_draw]

            dg.draw_graph(data, draw_choice)

            print("\n Press any key to continue ")


        elif instruction == 'e':
            print("Termination...\n")
            time.sleep(1)
            exit()
            #return

        input()



pr = cProfile.Profile()
pr.enable()

sampling = input("\n Sampling? (y/n/<Enter> default, no sampling): ")

if sampling == 'y':
    if num_of_users == 100:
        n = 50
    if num_of_users > 100:
        n = 100

    userBaskets = SampledApriori(n)

elif sampling == 'n':
    userBaskets = CreateMovieBaskets(min_rating)

else:
    userBaskets = CreateMovieBaskets(min_rating)


Apriori_params = input("\n Enter <min_frequency (0, 1), max Item_length> -OR- Press <Enter> for Default (0.15, 4): $ ")
splited_Apriori_params = Apriori_params.split(",")


if len(splited_Apriori_params) == 2:
    
    min_freq = float(splited_Apriori_params[0])
    max_items = int(splited_Apriori_params[1])

    t1_apriori = time.time()
    frequentPairs, supportTable = myApriori2(userBaskets, min_freq, max_items)
    t2_apriori = time.time()

# Default values for myApriori2
elif splited_Apriori_params[0] == '':
    t1_apriori = time.time()
    frequentPairs, supportTable = myApriori2(userBaskets)
    t2_apriori = time.time()

else:
    print(" BAD Input ")
    exit()


input_params = input("\n Enter Rule parameters \n[format: <min_confidence(0, 1), MinLift(-1 or >1), MaxLift(-1 or 0 < lift < 1)> -OR- Press <Enter> for Default(0.5, -1, -1)]: $" )
splited_params = input_params.split(",")

if len(splited_params) == 3:
    min_conf = float(splited_params[0])
    min_lift = float(splited_params[1])
    max_lift = float(splited_params[2])

    t1_rules = time.time()
    rules = AssociationRulesCreation(frequentPairs, supportTable, min_conf, min_lift, max_lift)
    t2_rules = time.time()

# Default values
elif splited_params[0] == '':
    t1_rules = time.time()
    rules = AssociationRulesCreation(frequentPairs, supportTable)
    t2_rules = time.time()

else:
    print(" BAD Input ")
    exit()

ta = t2_apriori - t1_apriori
tr = t2_rules - t1_rules

print("\n" * 10)
print("-" * 130)
print("| Apriori time = {}".format(ta))
print("| ===> {} rules discovered in {} sec".format(len(rules), tr))
print("| \n| Total execution time: {} sec\n".format(ta + tr))

presentRules(rules)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
# print(s.getvalue())

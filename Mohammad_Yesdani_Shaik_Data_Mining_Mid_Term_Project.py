# Libraries Used
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import timeit

# Function for finding possible Itemsets
def itemsetsFunc(items):
    itemsets = [[]]
    for item in items:
        new_itemsets = []
        for subset in itemsets:
            new_subset = subset + [item]
            new_itemsets.append(new_subset)
        itemsets.extend(new_itemsets)
    return itemsets[1:]

# Brute Force Function with Association Rules
def bruteFun(dataFrame, min_sup, confidence):
        # Calculate frequency of individual items
        trans = dataFrame["Transactions"].str.split(", ")
        num_trans = len(trans)
        subltt = []
        for sublt in trans:
            for item in sublt:
                subltt.append(item)
        item_frequency = pd.Series(subltt).value_counts()
    
        # Filtering the dataframe based on minimum support
        freq_df = item_frequency[item_frequency >= (min_sup / 100) * num_trans].reset_index()
        freq_df.columns = ["Itemsets", "Frequency"]
        freq_df["Support"] = (freq_df["Frequency"] / num_trans) * 100
    
        # Generating itemsets
        pos_itemsets = itemsetsFunc(freq_df["Itemsets"])
        itemsets = []
        for item_set in pos_itemsets:
            if len(item_set) > 1:
                itemsets.append(item_set)

        # Count of all itemsets
        itemset_counts = {}
        for it_set in itemsets:
            count = 0
            for transact in trans:
                if set(it_set).issubset(set(transact)):
                    count += 1
            itemset_counts[tuple(it_set)] = count
        itemset_support_df = pd.DataFrame(list(itemset_counts.items()), columns=["Itemsets", "Frequency"])
        itemset_support_df["Support"] = (itemset_support_df["Frequency"] / num_trans) * 100
    
        # Combining all frequent items and itemsets
        all_freq_items = pd.concat([freq_df, itemset_support_df[itemset_support_df["Support"] >= min_sup]], ignore_index=True)
    
        # Filtering the dataframe based on confidence
        tuple_lt = []
        for val in all_freq_items["Itemsets"]:
            if isinstance(val, str):
                tu = tuple([val])
                tuple_lt.append(tu)
            else:
                tuple_lt.append(val)
        all_freq_items["Itemsets"] = tuple_lt
        item_support_dict = dict(zip(all_freq_items["Itemsets"], all_freq_items["Support"]))
        final_dict = {}
        final_sup_lt = []
        for a in item_support_dict:
            for b in item_support_dict:
                if (set(b).issubset(a)) and (a != b):
                    conf = (item_support_dict[a] / item_support_dict[b]) * 100
                    final_dict[(a, b)] = conf
                    final_sup_lt.append(item_support_dict[a])
    
        # Final Manual Association Rules
        final_df = pd.DataFrame(final_dict.items(), columns=["Rules", "Confidence"])
        final_df["Support"] = final_sup_lt
        final_df = final_df[final_df["Confidence"] >= confidence].reset_index(drop=True)
        return final_df, all_freq_items

# Using Apriori Algorithm Library
def aprioriFun(dataFrame, min_sup, conf):
    transAp = dataFrame["Transactions"].str.get_dummies(sep=", ")
    transAp = transAp.astype(bool)
    frequent_itemsets = apriori(transAp, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=conf)
    frequent_itemsets = frequent_itemsets.rename(columns={"itemsets": "Itemsets", "support": "Support"})
    col_order = ["Itemsets", "Support"]
    frequent_itemsets = frequent_itemsets[col_order]
    frequent_itemsets["Support"] = frequent_itemsets["Support"] * 100
    return rules, frequent_itemsets

# Using FP-Tree Growth Algorithm Library
def fptreeFun(dataFrame, min_sup, conf):
    transFP = dataFrame["Transactions"].str.get_dummies(sep=", ")
    transFP = transFP.astype(bool)
    frequent_itemsets = fpgrowth(transFP, min_support=min_sup, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=conf)
    frequent_itemsets = frequent_itemsets.rename(columns={"itemsets": "Itemsets", "support": "Support"})
    col_order = ["Itemsets", "Support"]
    frequent_itemsets = frequent_itemsets[col_order]
    frequent_itemsets["Support"] = frequent_itemsets["Support"] * 100
    return rules, frequent_itemsets

# Printing Brute Force Final Association Rules
def printingFinalData(final_df):
    for i in range(len(final_df)):
        for col in final_df.columns:
            if col == "Rules":
                print(f"Rule_{i + 1}: {final_df.iloc[i][col][0]} --> {final_df.iloc[i][col][1]}", end=" ")
                continue
        print(f"Support: {final_df.iloc[i]["Support"]} Confidence: {final_df.iloc[i]["Confidence"]}\n")

# Printing Apriori and FP-Tree Growth Final Association Rules
def printingFinalData_lib(final_df):
    for j in range(len(final_df)):
        print(f"Rule_{j + 1}: {tuple(final_df.iloc[j]["antecedents"])}, {tuple(final_df.iloc[j]["consequents"])} --> {tuple(final_df.iloc[j]["consequents"])} Support: {final_df.iloc[j]["support"] * 100} Confidence: {final_df.iloc[j]["confidence"] * 100}\n")

# Start of the program
shop_company = {1: "Amazon", 2: "Best_Buy", 3: "K-mart", 4: "Nike", 5: "Generic", 6: "Custom", 7: "Quit"}

# Creating While Loop for finding Frequent Itemsets and Association Rules for different transactions.
while True:
    print("\nWelcome to Data Mining\n")
    for num, name in shop_company.items():
        print(f"{num}: {name}") 
    # Selecting Which Company Transactional Data to be Used.
    sel_comp = int(input("\nSelect a Shopping Company from above: "))
    if sel_comp == 7:
        print("Quitting.........")
        break
    print(f"\nYou have selected {sel_comp}: \"{shop_company[sel_comp]}\" shop company\n")
    
    # Giving Support and Confidence
    min_support = float(input("Minimum support between (1% to 100%): "))
    confidence = float(input("Confidence between (1% to 100%): "))

    if sel_comp in shop_company:
        # Load the selected company's dataset
        data_link = f"https://raw.githubusercontent.com/Yesdani20/Datasets/refs/heads/main/{shop_company[sel_comp]}_Dataset.csv"
        df = pd.read_csv(data_link)

        # Executing Brute Force Algorithm
        bruteForce_df, bruteForce_freqItems = bruteFun(df, min_support, confidence)
        print(f"\nFrequent Itemsets from \"Brute Force\" Algorithm:\n{bruteForce_freqItems}")
        print(f"\nManual Final Association Rules:")
        printingFinalData(bruteForce_df)

        min_support_u = min_support / 100
        confidence_u = confidence / 100

        # Executing Apriori Algorithm
        apriori_rules, apriori_freqItems = aprioriFun(df, min_support_u, confidence_u)
        print(f"\nFrequent Itemsets from \"Apriori\" Algorithm:\n{apriori_freqItems}")
        print(f"\nFinal Association Rules from \"Apriori\" Algorithm:")
        printingFinalData_lib(apriori_rules)

        # Executing FP-Tree Growth Algorithm
        fptree_rules, fptree_freqItems = fptreeFun(df, min_support_u, confidence_u)
        print(f"\nFrequent Itemsets from \"FP-Tree Growth\" Algorithm:\n{fptree_freqItems}")
        print(f"\nFinal Association Rules from \"FP-Tree Growth\" Algorithm:")
        printingFinalData_lib(fptree_rules)

        # Calculating Time Taken for each Algorithm
        print(f"Time Taken by \"Brute Force\" Algorithm: {timeit.timeit(lambda: bruteFun(df, min_support, confidence), globals=globals(), number=1)} seconds")
        print(f"Time Taken by \"Apriori\" Algorithm: {timeit.timeit(lambda: aprioriFun(df, min_support_u, confidence_u), globals=globals(), number=1)} seconds")
        print(f"Time Taken by \"FP-Tree Growth\" Algorithm: {timeit.timeit(lambda: fptreeFun(df, min_support_u, confidence_u), globals=globals(), number=1)} seconds")
    else:
        print(f"{sel_comp} is an invalid input. Please enter a valid number.")

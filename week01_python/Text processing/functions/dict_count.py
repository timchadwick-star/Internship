def dict_count(lst):
    count = {}
    for item in lst:
        if item in count:
            count[item]+=1
        else: count[item]=1
    return count

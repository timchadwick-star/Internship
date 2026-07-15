def list_stats(lst):
    lst_size=len(lst)
    sorted_lst=sorted(lst)
    lst_min=sorted_lst[0]
    lst_max=sorted_lst[-1]
    lst_sum=sum(lst)
    lst_mean=lst_sum/lst_size
    print('size: '+str(lst_size)+'\nmin: '+str(lst_min)+'\nmax: '+str(lst_max)+'\nsum: '+str(lst_sum)+'\nmean: '+str(lst_mean))
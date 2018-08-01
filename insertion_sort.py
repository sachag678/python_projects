
def insertion_sort(unsorted_list):
	i = 1
	while i <len(unsorted_list):
		j = i
		while j>0 and unsorted_list[j-1]>unsorted_list[j]:
			unsorted_list[j-1],unsorted_list[j] = unsorted_list[j],unsorted_list[j-1]
			j = j-1
		i = i+ 1

	return unsorted_list

l = [3,7,4,9,5,2,6,1]
print(insertion_sort(l))
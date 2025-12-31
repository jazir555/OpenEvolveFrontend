
# EVOLVE-BLOCK-START
def bubble_sort(arr):
    """Sort an array using bubble sort (inefficient implementation)"""
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
# EVOLVE-BLOCK-END

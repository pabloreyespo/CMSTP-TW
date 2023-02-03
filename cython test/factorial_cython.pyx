cpdef aux(int menor,int mayor):
    if menor +1 < mayor:
        mid = (menor+mayor)//2
        return aux(menor,mid)*aux(mid+1,mayor)
    if menor == mayor:
        return menor
    return menor*mayor

cpdef fact(int n):
    if n <= 1:
        return 1
    return aux(1,n)


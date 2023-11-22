times = ["60","120","180"]
nnodes = ["150","200"]
proporciones = [(20,20),(20,10),(15,10),(10,10)]

for time in times:
    for k in nnodes:
        for i,j in proporciones:
            print(f"python test_proporciones_prompt.py -i {i} -j {j} -k {k} -t {time}")
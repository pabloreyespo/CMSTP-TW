from datetime import datetime

inicio1 = datetime.strptime("2023-10-16 09:30:12", '%Y-%m-%d %H:%M:%S')
fin1 = datetime.strptime("2023-10-18 02:18:29", '%Y-%m-%d %H:%M:%S')
inicio2 = datetime.strptime("2023-10-18 16:54:54", '%Y-%m-%d %H:%M:%S')
fin2 = datetime.strptime("2023-10-19 18:37:21", '%Y-%m-%d %H:%M:%S')

segundos = (fin1-inicio1).total_seconds() + (fin2-inicio2).total_seconds()
horas = segundos / 3600
minutos = horas%1 * 60
segs = minutos%1 * 60

print(segundos)
print(horas)
print(minutos)
print(segs)

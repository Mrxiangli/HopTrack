import sys
import pprint
import xlsxwriter

# this script only works for Jetpack 4.6 

pp = pprint.PrettyPrinter(indent=4)

filename = sys.argv[1]

usage = open(filename, 'r')
lines = usage.readlines()

ct=0
total_ram = 0
ram_cap = 0
total_core_usage = 0
num_cores = 8
total_gpu_usage  = 0
for line in lines:
	tmp = line
	total_ram += int(tmp.split(' ')[1].split('/')[0])
	ram_cap = int(tmp.split(' ')[1].split('/')[1][0:5])
	cpu_cores = tmp.split(' ')[9].split('[')[1].split(']')[0].split(',')
	# cpu core averag usage calculation 
	core_tmp = 0
	for core in cpu_cores:
		#print(core.split('%@')[0])
		core_usage = int(core.split('%@')[0])
		core_tmp += core_usage
	core = core_tmp/num_cores
	ct += 1
	total_core_usage += core
	average_core = total_core_usage/ct
	
	gpu_cores = tmp.split(' ')[13]
	total_gpu_usage +=  int(gpu_cores.split('%@')[0])
	
usage_split = tmp.split(' ')
usage_dict = {}
usage_dict['Average RAM (MB)'] = 100*int(total_ram/ct)/ram_cap
print(usage_split)
for i, each in enumerate(usage_split):

	 if each == 'GPU':
	 	gpu_usage = usage_split[i+1]
	 	gpu_usage = gpu_usage.split('/')
	 	gpu_power = gpu_usage[1]
	 	usage_dict['Average GPU power (W)'] = int(gpu_power)/1000 
	 if each == 'CPU' and len(usage_split[i+1]) <=15:
	 	cpu_usage = usage_split[i+1]
	 	cpu_usage = cpu_usage.split('/')
	 	cpu_power = cpu_usage[1]
	 	print(cpu_power)
	 	usage_dict['Average CPU power (W)'] = int(cpu_power)/1000 
usage_dict['CPU energy (J)'] = usage_dict['Average CPU power (W)']*ct
usage_dict['GPU energy (J)'] = usage_dict['Average GPU power (W)']*ct
usage_dict['CPU energy (KWh)'] = usage_dict['Average CPU power (W)']*ct*2.8*10**(-7)
usage_dict['GPU energy (KWh)'] = usage_dict['Average GPU power (W)']*ct*2.8*10**(-7)
usage_dict['Average CPU usage (%)'] = (average_core)
usage_dict['Average GPU usage (%)'] = (total_gpu_usage/ct)
out = sys.argv[1].split('.')[0]+"_data_new.xlsx"


workbook = xlsxwriter.Workbook(out)
 
worksheet = workbook.add_worksheet()
 
for i, key in enumerate(usage_dict.keys()):
	worksheet.write(i,0, key)
	worksheet.write(i,1, usage_dict[key])

workbook.close()


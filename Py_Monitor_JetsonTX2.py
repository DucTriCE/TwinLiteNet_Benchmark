#Author : Siddharth Menon (Written by reffering matplotlib documentation)
#Code specific to NVIDIA Jetson TX2
import sys 
import os
import json
import threading
from collections import deque
import time

import numpy as np
import statistics  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

class Analyser():
	t1 = threading.Thread
	start=False
	temporary_file = open("temporary_file.txt", "w+")
	
	def start_recording(self):
		self.t1 = threading.Thread(target=self.model_report) 
		self.t1.start()
		self.start = True
		
	def stop_recording(self):
		self.start = False
		self.t1.join()
 
	def model_report(self):
		gpu_list, cpu_list, cpu_list1, gpu_temp, cpu_temp, timer, list1, power_consumption_list = [], [], [], [], [], [], [], []
		
		def gpu_usage():	
			gpuLoadFile="/sys/devices/gpu.0/load"
			with open(gpuLoadFile, 'r') as gpuFile:
				gpuusage=gpuFile.read()
			gpu_value=float(gpuusage)/10
			return gpu_value

		def cpu_usage():
			cpuLoadFile="/proc/meminfo"
			cpu_free = 0
			with open(cpuLoadFile) as fp:
				for i, line in enumerate(fp):
					if i == 2:
						#cpu_free=((8-((float(line[17])*10 + float(line[18]))/10))/8)*100
						break
			return cpu_free
			
		def gpu_temperature():
			tempLoadFile="/sys/devices/virtual/thermal/thermal_zone2/temp"
			with open(tempLoadFile,'r') as tempFile:
				temp=tempFile.read()
			gpu_temp_value=float(temp)/1000
			return gpu_temp_value

		def cpu_temperature():
			tempLoadFile1="/sys/devices/virtual/thermal/thermal_zone1/temp"
			with open(tempLoadFile1,'r') as tempFile1:
				temp1=tempFile1.read()
			cpu_temp_value=float(temp1)/1000
			return cpu_temp_value
			
		def power_consumption():
			tempLoadFile2="/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power0_input"
			with open(tempLoadFile2,'r') as tempFile2:
				temp2=tempFile2.read()
			power_value=float(float(temp2)/1000)
			return power_value
			
		while(True):
			if(self.start==True):
				break
			
		start_time=time.time()	
		i=0
		with open('temporary_file.txt', 'w+') as temporary_file_write:
			while (self.start):
				i=i+6
				temporary_file_write.write("%f\n%f\n%f\n%f\n%f\n%f\n"%(gpu_usage(),cpu_usage(),gpu_temperature(),cpu_temperature(),power_consumption(),time.time()-start_time))
				timer.append(time.time()-start_time)
			
		file= open("temporary_file.txt","r")
		list1 = []
		value1=file.readlines()
		for value in value1:
			value=float(value)
			list1.append(value)
                
		i=0
		while(i<=len(list1)-6):
			gpu_list.append(list1[i])
			cpu_list.append(list1[i+1])
			gpu_temp.append(list1[i+2])
			cpu_temp.append(list1[i+3])
			power_consumption_list.append(list1[i+4])
			# ttime = 
			i=i+6	
		# with open(r'/home/ceec/TwinLiteNet_board/power_plot/timer_large.txt', 'w') as fp:
		# 	for item in timer:
		# 		fp.write("%f\n" % item)

		with open(r'/home/ceec/TwinLiteNet_done/power.txt', 'w') as fp:
			for item in power_consumption_list:
				fp.write("%f\n" % item)

		# plt.plot(timer, gpu_temp,label='GPU Temp')
		# plt.plot(timer,cpu_temp,label='CPU Temp')
		
		# plt.ylabel('Temperature')
		# plt.xlabel('Time (s)')
		# plt.title('Avg GPU temp: %.2f , Avg CPU Temp: %.2f'%(round(statistics.mean(cpu_temp), 2),round(statistics.mean(gpu_temp), 2)))
		# plt.legend()
		# plt.savefig('./temp_model.png')
		# plt.show()

		# plt.clf()
		# df = pd.DataFrame({'Power_Consumption': power_consumption_list})
		# ema_span = 200
		# df1 = df.ewm(span=ema_span).mean()

		# plt.plot(timer,df1['Power_Consumption'])
		# plt.xlabel('Time(s)')
		# plt.ylabel('Power consumption')
		# # print(len(power_consumption_list))
		# plt.title('AVG Power consumed: %.2f Watts'%round(statistics.mean(power_consumption_list), 2))
		
			
		# plt.savefig('./power_plot/power.png')

		# plt.show()


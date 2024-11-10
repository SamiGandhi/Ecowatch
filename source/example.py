import os
from parameters import parameters as para
import main


filename_ = "Wood_Stork.mp4"

para.initialize(video_path=filename_, HUMIDITY = 76.33, VEGETATION = 0.53,runing='coding', DISTANCE = Distance_vector[0], scale_factor = scale_factors[0],QF=90, ENVIRONEMENT= 'None')
main.run()
para.initialize(video_path=filename_, HUMIDITY = 76.33, VEGETATION = 0.53,runing='decoding',DISTANCE = Distance_vector[0], scale_factor = scale_factors[0], QF = 90, ENVIRONEMENT= 'None')
main.run()

para.initialize(video_path=filename_, HUMIDITY = 76.33, VEGETATION = 0.53,runing='Roi',DISTANCE = Distance_vector[0], scale_factor = scale_factors[0],H= 50, L=20 )
main.run()
para.initialize(video_path=filename_, HUMIDITY = 76.33, VEGETATION = 0.53,runing='decoding_roi',DISTANCE = Distance_vector[0], scale_factor = scale_factors[0],H= 50, L=20 )
main.run()
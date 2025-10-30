Section 2 is a vehicle inspecting a pipeline

"*Each vehicle is deployed in a cylinder of approximately 16m of height and operates on an elevation mechanism that exhibits a **spring-like effect**
when the vehicles stop at a designated vertical position*"

"once positioned at a given height, can rotate to perform visual inspections"

"One of the vehicles starts its motion from the base of the pipeline, while the other starts from its top"



-> bit confusing as to wheter they start at the middle or the extremities, from the data, it should be the extremity
-> from my understanding, one start top, other bottom, ghe center line they mean the vertical one 



➢ Plot the position, speed and acceleration of each vehicle as a function of time (total of 6 plots).
-> Accelerometer-and-Gyroscope-Analysis\plot_Section_II

➢ Plot the angular position and angular rate for each vehicle as a function of time (total of 4 plots).
->  done

➢ Determine which vehicle is going up and which vehicle is going down. Justify.
-> Vehicul 1 : 
        - Accelerometer-and-Gyroscope-Analysis\plot_Section_II\vehicle_1_pz.png
                - we can see its going **down** by about 20m~  so V1 is going from top to bottom, 
        - Accelerometer-and-Gyroscope-Analysis\plot_Section_II\vehicle_2_pz.png
                - we can see its going **up** by about 10m  so V1 is going from bottom to top,

➢ Estimate the vertical position (height) at which each vehicle stops to perform visual inspections, if
applicable.
        - v1 : -20m  with a first stop arround -8m, then a second at -19m (the max height is 16, so the extra 3m is probably bias and noise)
        - v2 : +10m , so from bottom to top , there is also a slight down to 8, then back to 10m, 


➢ Have both vehicles traveled the entire length of the pipeline? Justify.
        - v1 yes~ probably, since its overshooted the lenght, >16m
        - v2 no, it doesnt not seem to reach the 16m based on the calculated position

➢ For each stop of each vehicle in the vertical motion, if applicable, which vehicle turned right and
which vehicle turned left? Assume that a positive angular position represents the vehicle turning left, and a negative angular position represents turning right. 
        - v1 has turn at the midpoint of 3.15 rad, which is closer to 177~ +180 degrees -> which is turning left
        - v2 has turned at the midpoint of -6.25 rad ~ -360 degrees, towards the right 



➢ For all the turns you identified, did the vehicles perform a full 360-degree inspection? Justify.
        - v1 has done half a turn
        - v2 has seem to have done a full turn
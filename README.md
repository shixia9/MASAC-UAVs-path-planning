UAVs path planning algorithm based on multi-agent deep reinforcement learning algorithm MASAC, this version removes the definition of "**LEADER-FOLLOWER**" and considers all Uavs as **LEADERS**.
The main tasks of a UAV:

+ Avoid collision with obstacles
+ Avoid collisions with other UAVs
+ Reach its goal

The task can be described as the following figure:

![task-model](https://github.com/shixia9/MASAC-UAVs-path-planning/blob/master/imgs/fig1.png)

>Another version is the "LEADER-FOLLOWER" task model, see:
>
>[shixia9/MASAC-UAVs-path-planning-LF (github.com)](https://github.com/shixia9/MASAC-UAVs-path-planning-LF)

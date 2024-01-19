# export ROS_IP=$(hostname -I)

# if [ "$1" ]; then
# 	export ROS_MASTER_URI=http://$1:11311
# 	export ROS_HOSTNAME=$ROS_IP
	
# 	echo "ROS_IP:           $ROS_IP"
# 	echo "ROS_MASTER_URI:   $ROS_MASTER_URI"
# 	echo "ROS_HOSTNAME:     $ROS_HOSTNAME"
# else 
#         echo "missing MASTER IP"
# fi
## ESU connetion
#export ROS_MASTER_URI=http://192.168.135.200:11311
#export ROS_HOSTNAME=192.168.135.110
#export ROS_IP=192.168.128.198

## S_H_net connection
# Master IP address
#this is the ip of the ros master
export ROS_MASTER_URI=http://192.168.135.200:11311 
# Local computer IP Address
#this the ip of this pc, find it with 'ip a'
export ROS_HOSTNAME=192.168.135.200
#same as before 
export ROS_IP=192.168.135.200
echo "ROS_IP:           $ROS_IP"
echo "ROS_MASTER_URI:   $ROS_MASTER_URI"
echo "ROS_HOSTNAME:     $ROS_HOSTNAME"
# Where the setup.bash is
source /home/irong/bfmc_2022/sparcs/devel/setup.bash


#pragma once
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include "ros/ros.h"
#include "utils/pose.h"

namespace gazebo
{
    namespace pose
    {   
        class POSE: public ModelPlugin
    	{
        private: 
            physics::ModelPtr m_model;
            ros::NodeHandlePtr nh;
            ros::Timer timer;
	        /** ROS INTEGRATION **/
            // A node use for ROS transport
            std::unique_ptr<ros::NodeHandle> m_ros_node;
            // A ROS publisher
            ros::Publisher m_pubPOSE;
            // The gps message
            utils::pose m_pose;
        // Default constructor
        public: POSE();
        public: void Load(physics::ModelPtr, sdf::ElementPtr);
        public: void OnUpdate();        
        };
    };    
};
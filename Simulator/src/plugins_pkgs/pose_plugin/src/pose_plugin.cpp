
#include "pose_plugin.hpp"

#define DEBUG false

namespace gazebo
{
    namespace pose
    {   
        POSE::POSE():ModelPlugin() {}
     		
        void POSE::Load(physics::ModelPtr model_ptr, sdf::ElementPtr sdf_ptr)
        {
          nh = boost::make_shared<ros::NodeHandle>();
          timer = nh->createTimer(ros::Duration(0.001), std::bind(&POSE::OnUpdate, this));

  			  // Save a pointer to the model for later use
  			  this->m_model = model_ptr;
  			
        	// Create topic name      	
        	std::string topic_name = "/automobile/pose";
  	        
          // Initialize ros, if it has not already been initialized.
    			if (!ros::isInitialized())
    			{
      			  int argc = 0;
      			  char **argv = NULL;
      			  ros::init(argc, argv, "poseNODEvirt", ros::init_options::NoSigintHandler);
    			}

          this->m_ros_node.reset(new ::ros::NodeHandle("/poseNODEvirt"));

        	this->m_pubPOSE = this->m_ros_node->advertise<utils::pose>(topic_name, 2);
        	
          if(DEBUG)
          {
              std::cerr << "\n\n";
              ROS_INFO_STREAM("====================================================================");
              ROS_INFO_STREAM("[pose_plugin] attached to: " << this->m_model->GetName());
              ROS_INFO_STREAM("[pose_plugin] publish to: "  << topic_name);
              ROS_INFO_STREAM("====================================================================\n\n");
          }
        }

        // Publish the updated values
        void POSE::OnUpdate()
        {
		      this->m_pose.t     = this->m_model->GetWorld()->SimTime().Float();
          this->m_pose.x     = this->m_model->WorldPose().Pos().X();
          this->m_pose.y     = this->m_model->WorldPose().Pos().Y();
          this->m_pose.z     = this->m_model->WorldPose().Pos().Z();
          this->m_pose.vx    = this->m_model->WorldLinearVel().X();
          this->m_pose.vy    = this->m_model->WorldLinearVel().Y();
          this->m_pose.vz    = this->m_model->WorldLinearVel().Z();
          this->m_pose.ax    = this->m_model->WorldLinearAccel().X();
          this->m_pose.ay    = this->m_model->WorldLinearAccel().Y();
          this->m_pose.az    = this->m_model->WorldLinearAccel().Z();
          this->m_pose.phi   = this->m_model->WorldPose().Rot().Roll();
          this->m_pose.teta  = this->m_model->WorldPose().Rot().Pitch();
          this->m_pose.psi   = this->m_model->WorldPose().Rot().Yaw();
          this->m_pose.vphi  = this->m_model->WorldAngularVel().X();
          this->m_pose.vteta = this->m_model->WorldAngularVel().Y();
          this->m_pose.vpsi  = this->m_model->WorldAngularVel().Z();
          this->m_pose.aphi  = this->m_model->WorldAngularAccel().X();
          this->m_pose.ateta = this->m_model->WorldAngularAccel().Y();
          this->m_pose.apsi  = this->m_model->WorldAngularAccel().Z();
          //publish
          this->m_pubPOSE.publish(this->m_pose);
        };      
    }; //namespace trafficLight
    GZ_REGISTER_MODEL_PLUGIN(pose::POSE)
}; // namespace gazebo
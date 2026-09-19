#!/usr/bin/env python3
"""
Test and validate DRL navigation system
Checks all components before deployment
"""

import rospy
import sys
import os

def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("Checking Python dependencies...")
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'gym': 'OpenAI Gym',
        'rospkg': 'ROS Python',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_ros_setup():
    """Check ROS environment"""
    print("\nChecking ROS setup...")
    
    # Check if ROS is sourced
    if 'ROS_PACKAGE_PATH' not in os.environ:
        print("  ✗ ROS environment not sourced")
        return False
    print("  ✓ ROS environment sourced")
    
    # Check if autonomous_flight package exists
    try:
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('autonomous_flight')
        print(f"  ✓ autonomous_flight package found at: {pkg_path}")
        return True
    except Exception as e:
        print(f"  ✗ autonomous_flight package not found: {e}")
        return False


def check_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    
    import rospkg
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('autonomous_flight')
    
    model_path = os.path.join(pkg_path, 'models', 'drl_tunnel_nav.pth')
    if os.path.exists(model_path):
        print(f"  ✓ Model file found: {model_path}")
        return True
    else:
        print(f"  ⚠ Model file not found: {model_path}")
        print("    You need to train a model first or download a pre-trained one")
        return False


def test_drl_agent():
    """Test DRL agent initialization"""
    print("\nTesting DRL agent...")
    
    try:
        # Add scripts directory to path
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('autonomous_flight')
        sys.path.insert(0, os.path.join(pkg_path, 'scripts'))
        
        from drl_agent import DRLNavigationAgent
        import numpy as np
        
        # Create agent
        agent = DRLNavigationAgent(state_dim=21, action_dim=4, device='cpu', model_path=None)
        print("  ✓ DRL agent initialized")
        
        # Test inference
        dummy_state = np.random.randn(21)
        action = agent.get_action(dummy_state)
        
        if action.shape == (4,):
            print(f"  ✓ Agent inference working")
            print(f"    Sample action: forward={action[0]:.2f} m/s, "
                  f"lateral={action[1]:.2f} m/s, "
                  f"vertical={action[2]:.2f} m/s, "
                  f"yaw_rate={action[3]:.2f} rad/s")
            return True
        else:
            print(f"  ✗ Wrong action shape: {action.shape}")
            return False
            
    except Exception as e:
        print(f"  ✗ DRL agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gym_environment():
    """Test Gym environment"""
    print("\nTesting Gym environment...")
    
    try:
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('autonomous_flight')
        sys.path.insert(0, os.path.join(pkg_path, 'scripts'))
        
        from tunnel_gym_env import TunnelNavigationEnv
        import numpy as np
        
        # Create environment (without ROS)
        env = TunnelNavigationEnv(use_ros=False)
        print("  ✓ Gym environment created")
        
        # Test reset
        state = env.reset()
        if state.shape == (21,):
            print(f"  ✓ Environment reset working (state shape: {state.shape})")
        else:
            print(f"  ✗ Wrong state shape: {state.shape}")
            return False
        
        # Test step
        action = np.random.uniform(-1, 1, size=4)
        next_state, reward, done, info = env.step(action)
        print(f"  ✓ Environment step working (reward: {reward:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Gym environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config_files():
    """Check if configuration files exist"""
    print("\nChecking configuration files...")
    
    import rospkg
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('autonomous_flight')
    
    config_files = [
        'cfg/dynamic_navigation/drl_param.yaml',
        'cfg/dynamic_navigation/drl_flight_base.yaml',
        'cfg/dynamic_navigation/data_recorder_param.yaml',
    ]
    
    all_exist = True
    for config_file in config_files:
        full_path = os.path.join(pkg_path, config_file)
        if os.path.exists(full_path):
            print(f"  ✓ {config_file}")
        else:
            print(f"  ✗ {config_file} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_launch_files():
    """Check if launch files exist"""
    print("\nChecking launch files...")
    
    import rospkg
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('autonomous_flight')
    
    launch_file = os.path.join(pkg_path, 'launch', 'drl_dynamic_navigation.launch')
    if os.path.exists(launch_file):
        print(f"  ✓ drl_dynamic_navigation.launch")
        return True
    else:
        print(f"  ✗ drl_dynamic_navigation.launch - NOT FOUND")
        return False


def main():
    print("="*60)
    print("DRL Navigation System Validation")
    print("="*60)
    print()
    
    results = {}
    
    # Run checks
    results['python_deps'], missing = check_python_dependencies()
    results['ros_setup'] = check_ros_setup()
    results['config_files'] = check_config_files()
    results['launch_files'] = check_launch_files()
    results['model_files'] = check_model_files()
    results['drl_agent'] = test_drl_agent()
    results['gym_env'] = test_gym_environment()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check.upper():20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ System is ready for deployment!")
        print("\nNext steps:")
        print("1. Train a model: python3 scripts/train_drl.py")
        print("2. Launch navigation: roslaunch autonomous_flight drl_dynamic_navigation.launch")
        return 0
    elif results['python_deps'] and results['ros_setup']:
        print("\n⚠ System is partially ready")
        if not results['model_files']:
            print("\nMissing trained model. Run:")
            print("  python3 scripts/train_drl.py --total_timesteps 100000")
        return 1
    else:
        print("\n✗ Critical issues found. Please fix errors above.")
        if not results['python_deps']:
            print("\nInstall missing dependencies:")
            print("  pip3 install -r requirements.txt")
        if not results['ros_setup']:
            print("\nSource ROS workspace:")
            print("  source ~/catkin_ws/devel/setup.bash")
        return 2


if __name__ == '__main__':
    sys.exit(main())

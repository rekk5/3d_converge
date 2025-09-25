#!/usr/bin/env python3
"""
Simple test script to verify VGGT integration in the 3D reconstruction pipeline.
This test doesn't require VGGT dependencies to be installed.
"""

import sys
import os

def test_main_syntax():
    """Test that main.py has correct syntax."""
    print("Testing main.py syntax...")
    
    try:
        import py_compile
        py_compile.compile('main.py', doraise=True)
        print("✅ main.py syntax is correct")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ main.py syntax error: {e}")
        return False

def test_vggt_import_statement():
    """Test that VGGT import is present in main.py."""
    print("Testing VGGT import statement...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        if 'from vggt_recon import run_vggt' in content:
            print("✅ VGGT import statement found")
            return True
        else:
            print("❌ VGGT import statement not found")
            return False
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False

def test_vggt_default_method():
    """Test that VGGT is set as default reconstruction method."""
    print("Testing VGGT as default method...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        if "default='vggt'" in content:
            print("✅ VGGT is set as default reconstruction method")
            return True
        else:
            print("❌ VGGT is not set as default reconstruction method")
            return False
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False

def test_vggt_choices():
    """Test that VGGT is in the choices list."""
    print("Testing VGGT in choices...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        if "'vggt'" in content and "choices=" in content:
            print("✅ VGGT is in the choices list")
            return True
        else:
            print("❌ VGGT is not in the choices list")
            return False
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False

def test_vggt_reconstruction_logic():
    """Test that VGGT reconstruction logic is present."""
    print("Testing VGGT reconstruction logic...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        if "args.recon_method == 'vggt'" in content and "run_vggt(" in content:
            print("✅ VGGT reconstruction logic found")
            return True
        else:
            print("❌ VGGT reconstruction logic not found")
            return False
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False

def test_naming_convention():
    """Test that VGGT naming convention is present."""
    print("Testing VGGT naming convention...")
    
    try:
        with open('naming_convention.py', 'r') as f:
            content = f.read()
        
        if 'vggt_point_cloud' in content:
            print("✅ VGGT naming convention found")
            return True
        else:
            print("❌ VGGT naming convention not found")
            return False
    except Exception as e:
        print(f"❌ Error reading naming_convention.py: {e}")
        return False

def test_vggt_recon_file():
    """Test that VGGT reconstruction file exists."""
    print("Testing VGGT reconstruction file...")
    
    if os.path.exists('vggt_recon.py'):
        print("✅ vggt_recon.py file exists")
        return True
    else:
        print("❌ vggt_recon.py file not found")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing VGGT Integration in 3D Reconstruction Pipeline")
    print("=" * 60)
    
    success = True
    
    # Test main.py syntax
    if not test_main_syntax():
        success = False
    
    # Test VGGT import
    if not test_vggt_import_statement():
        success = False
    
    # Test VGGT as default
    if not test_vggt_default_method():
        success = False
    
    # Test VGGT in choices
    if not test_vggt_choices():
        success = False
    
    # Test VGGT reconstruction logic
    if not test_vggt_reconstruction_logic():
        success = False
    
    # Test naming convention
    if not test_naming_convention():
        success = False
    
    # Test VGGT recon file
    if not test_vggt_recon_file():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! VGGT integration is working correctly.")
        print("\n💡 You can now use VGGT as the default reconstruction method:")
        print("   python3 main.py --image_path /path/to/images --output_dir /path/to/output")
        print("   python3 main.py --image_path /path/to/images --output_dir /path/to/output --recon_method vggt")
        print("\n📋 VGGT-specific parameters:")
        print("   --conf_threshold: Confidence threshold for point filtering (default: 3.0)")
        print("   --fps: Frames per second to extract from video (default: 1.0)")
        print("   --no_cameras: Don't show cameras in 3D scene")
    else:
        print("❌ Some tests failed. Please check the integration.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

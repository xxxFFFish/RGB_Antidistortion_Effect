# RGB_Antidistortion_Effect
This project contains a series of Python scripts which used for baking map textures and a Godot project which used for demonstrating RGB antidistortion effect.

It is recommended to run this project using Python 3.x and Godot 4.5 version.

These data models are applicable to the optical design of VR or AR.

The data of "bake_script/raw_data/distortion_raw_data_r/g/b.csv" should come from an optical design software (e.g. ZEMAX).

User can set the variates such as screen resolution, screen pixel scale, models and format based on the actual situation. User can use the virtual data and setting variates in this project to explore its effect if they does not have a specific design.

User can first run the script of bake_scrit/quick_start.py to complete the baking and copying of map textures when they runs this project for the first time. However, it should be noted that the PNG files copied into the Godot project need to ensure that they are imported as Image!

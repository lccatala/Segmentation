# Segmentation
Trying to replicate Apple's thing where you can press and hold an object in a photo and it selects it for you.
For example
![Subject lifting in iOS of a focaccia sandwich](images/goal.png "Goal")

## Instructions (so far)
Download the DeeplabV3 model from pytorch hub and convert it to Apple's format
1. Create a pip environment 
`python3 -m venv .venv` 
2. Activate it
`source .venv/bin/activate` 
3. Install requirements
`pip install -r requirements.txt`  
4. Run the script for downloading and converting
`python3 models/convert.py`

# Hand gesture controlled video game

This program is mainly for car/racing games which only utilises WASD keys.

main_UI.py has some UI elements and a proper interface

main_noUI.py is simple camera window with a skeleton and no interface

~~Datasets are from kaggle, you can change the dataset as per your requirement or put your own, just replace them in models directory~~
GitHub size limit didn't allow me to push the datasets, so get your own from kaggle.
The one I used was- https://www.kaggle.com/models/odayahmed/sign-language-models

## The main controls and gestures are- 
- 'palm': 'w',        # Forward
- 'fist': 's',        # Backward
- 'right': 'd',       # Right (1 finger)
- 'left': 'a',        # Left (2 fingers)
- '3_fingers': 'space' # Space (brake)

## How to use a dataset: 
  - Download any hand sign recognition dataset(or use the one I provided above)
  - Create a directory named "models" in the main directory
  - Put the dataset in that folder
  - search for this line in the main py file `model_path = 'models/Hypermodel.h5'`
  - Rename the `Hypermodel.h5` to your dataset's name
  - Save and execute

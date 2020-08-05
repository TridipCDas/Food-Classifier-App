# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "Food-11"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

# set the path to the serialized model after training
MODEL_PATH = "food11.model"

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# initialize the list of class label names
CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
	"Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
	"Vegetable/Fruit"]

# set the batch size when fine-tuning
BATCH_SIZE = 32


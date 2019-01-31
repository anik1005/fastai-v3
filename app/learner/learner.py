 BASE_PATH = './'
EXAMPLE_PATH = './example.jpg'

# Imports
from fastai.vision import *
from fastai.widgets import *

# Set the file path
path = Path(BASE_PATH)

# Create data loader / manager
data_bunch = ImageDataBunch.from_folder(
    # Data director
    path, 
    # Reserve 20 percent of our images for our validation set
    valid_pct=0.2,
    # The transforms to use (convenience method for us)
    ds_tfms=get_transforms(max_zoom=1.0), 
    # Dimension of image to process
    size=224,
    # Num workers to use
    num_workers=4
).normalize(
	# Use imagenet stats to normalize (to match what was pre-trained with)
	imagenet_stats
)

# Create our learner to process the training data and update our model
learner = create_cnn(data_bunch, models.resnet34, pretrained=True, metrics=error_rate)	

# Start training
learner.fit_one_cycle(10)

# Unfreeze the model
learner.unfreeze()

# Train the entire model some more
learner.fit_one_cycle(10, max_lr=slice(1e-3, 1e-5))

# Reduce the learning rate and train some more
learner.fit_one_cycle(10, max_lr=slice(1e-3, 1e-5))

# Grab an example image
example_image = open_image(Path(EXAMPLE_PATH))

# Make a prediction
predicted_class, _, _ = learner.predict(example_image)
print(predicted_class)
from basic_processing import run_basic_processing
from huggingface_loader import load_food_image
from preprocessing import save_preprocessed_images


run_basic_processing()

food_image = load_food_image()
save_preprocessed_images(food_image)
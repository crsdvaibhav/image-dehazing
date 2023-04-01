# Image DeBlurring API

This is an Image DeBlurring API written using PyTorch and FastAPI. It is based on the follwing research paper: [DeblurGAN](https://arxiv.org/abs/1711.07064)

## Details

The API has 3 endpoints:
- `/upload` This endpoints gets the image from the frontend, checks whether it is jpg, png or jpeg. If not it throws an error message. If it is a valid image it stores it,
- `/predict` This endpoint takes in the image and sends it through the generator, saves the deblurred image, and sends it back.
- `/clear` This endpoint clears both the uploaded and deblurred image from the server.

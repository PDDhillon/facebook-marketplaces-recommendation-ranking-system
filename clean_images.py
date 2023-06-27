
from PIL import Image
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def clean_image_data(filepath):
    """Resizes images based on provided filepath
    
    Parameters:
    filepath (string): A filepath containing images
      
    """
    image_dir = os.listdir(filepath)
    os.mkdir(f'{filepath}/cleaned_images')
    final_size = 512
    for item in image_dir:
        im = Image.open(f'{filepath}/{item}')
        new_im = resize_image(final_size, im)
        new_im.save(f'{filepath}/cleaned_images/{item}','JPEG')

if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    final_size = 512
    for n, item in enumerate(dirs[:5], 1):
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{n}_resized.jpg')

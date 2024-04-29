from PIL import Image, ImageSequence

def mirror_flip(image, l2r = True):
    width, height = image.size
    half_width = width // 2

    left_half = image.crop((0, 0, half_width, height))
    right_half = image.crop((half_width, 0, width, height))

    if l2r:
        mirrored_right = left_half.transpose(Image.FLIP_LEFT_RIGHT)
        new_image = Image.new('RGB', (width, height))
        new_image.paste(left_half, (0, 0))
        new_image.paste(mirrored_right, (half_width, 0))
    
    else:
        mirrored_left = right_half.transpose(Image.FLIP_LEFT_RIGHT)
        new_image = Image.new('RGB', (width, height))
        new_image.paste(right_half, (half_width, 0))
        new_image.paste(mirrored_left, (0, 0))

    return new_image

def mirror_flip_image(input_path, output_path, l2r=True):
    image = Image.open(input_path)
    new_image = mirror_flip(image, l2r)
    new_image.save(output_path)

def mirror_flip_gif(input_path, output_path, l2r=True):
    with Image.open(input_path) as gif:
        frames = []

        for frame in ImageSequence.Iterator(gif):
            new_frame = mirror_flip(frame, l2r)
            frames.append(new_frame)

        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)

input_gif_path = 'target.gif'
output_gif_path = 'output.gif'
input_jpg_path = '1.jpg'
output_jpg_path = 'output.jpg'

#mirror_flip_gif(input_gif_path, output_gif_path, l2r=False)
#mirror_flip_image(input_jpg_path, output_jpg_path, l2r=True)

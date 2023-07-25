from torchvision import transforms
class ImageProcessor:
    def __init__(self):
        pass

    def process(self, image):
        #apply tramsformations        
         transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
        return image

